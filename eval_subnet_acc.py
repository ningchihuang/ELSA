# --------------------------------------------------------
# TinyViT Main (train/validate)
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Add distillation with saved teacher logits
# --------------------------------------------------------

import os
import time
import random
import argparse
import datetime
from collections import defaultdict
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
from my_meter import AverageMeter

from config import get_config
from models import build_model
from data import build_loader, build_proxy_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint,\
    NativeScalerWithGradNormCount,\
    auto_resume_helper, is_main_process,\
    add_common_args,\
    get_git_info

from nas_utils import LinearEpsilonScheduler, CandidatePool, RandomCandGenerator
from models.remap_layer import RemapLayer
remap_layer_22kto1k = RemapLayer('./imagenet_1kto22k.txt')

try:
    import wandb
except ImportError:
    wandb = None
NORM_ITER_LEN = 100


def parse_option():
    parser = argparse.ArgumentParser(
        'TinyViT training and evaluation script', add_help=False)
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(args, config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config)
    
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    
    model.set_sampler_info(config.NAS.SEARCH_SPACE)
    logger.info("Register search space")
    logger.info(f"*FLOPs of largest subnetworks: {model.max_flops}")
    logger.info(f"*FLOPs of smallest subnetworks: {model.min_flops}")
    
    if not args.only_cpu:
        model.cuda()

    if args.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if not args.only_cpu:
        model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters = False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
        

    
    if config.MODEL.PRETRAINED is None:
        raise ValueError("Please provide pretrained path")
    
    load_pretrained(config, model_without_ddp, logger)
    
    # Set the largest by default
    if config.NAS.INIT_CONFIG is None:
        raise ValueError("Please set the NAS.INIT_CONFIG")
    else:
        model_without_ddp.set_sample_config(config.NAS.INIT_CONFIG)
    
    logger.info(str(model_without_ddp))
    
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops}")
    
    acc1, acc5, loss = validate(args, config, data_loader_val, model)
    logger.info(
        f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")


def is_valid_grad_norm(num):
    if num is None:
        return False
    return not bool(torch.isinf(num)) and not bool(torch.isnan(num))


def set_bn_state(config, model):
    if config.TRAIN.EVAL_BN_WHEN_TRAINING:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()

@torch.no_grad()
def validate(args, config, data_loader, model, num_classes=1000):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        if not args.only_cpu:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)
        if num_classes == 1000:
            output_num_classes = output.size(-1)
            if output_num_classes == 21841:
                output = remap_layer_22kto1k(output)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    acc1_meter.sync()
    acc5_meter.sync()
    logger.info(
        f' The number of validation samples is {int(acc1_meter.count)}')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    args, config = parse_option()
    config.defrost()
    if config.DISTILL.TEACHER_LOGITS_PATH:
        config.DISTILL.ENABLED = True
    config.freeze()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    if args.only_cpu:
        ddp_backend = 'gloo'
    else:
        torch.cuda.set_device(config.LOCAL_RANK)
        ddp_backend = 'nccl'

    torch.distributed.init_process_group(
        backend=ddp_backend, init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(config.SEED)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if is_main_process():
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

        config_dict = dict(config)
        config_dict['git'] = get_git_info()
        if args.use_wandb:
            wandb_output_path = config.OUTPUT
            wandb.init(project="sparsity", entity='max410011', config=config_dict,
                       dir=wandb_output_path, name=args.cfg[:-4] + "_" + args.tag)

    # print git info
    logger.info('===== git =====')
    logger.info(str(get_git_info()))

    # print config
    logger.info(config.dump())

    main(args, config)
