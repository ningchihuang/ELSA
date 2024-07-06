import os
import time
import random
import argparse
import datetime
from collections import defaultdict
import numpy as np
import json

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
from my_meter import AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint,\
    NativeScalerWithGradNormCount,\
    auto_resume_helper, is_main_process,\
    get_git_info, run_cmd, add_common_args



def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script', add_help=False)
    add_common_args(parser)
    # random sample parameter
    parser.add_argument('--param-limits', type=float, default=2.2)
    parser.add_argument('--min-param-limits', type=float, default=1.6)
    parser.add_argument('--sample_num', type=int, default=100, help='number of subnet to sample')
    
    args = parser.parse_args()

    config = get_config(args)

    return args, config

import copy
from nas_utils import RandomCandGenerator


class SubnetSampler():
    def __init__(self, args, model, model_without_ddp, search_space, val_loader, config):
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        self.val_loader = val_loader
        self.memory = []
        self.vis_dict = {}
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.rcg = RandomCandGenerator(search_space)
        self.config = config
        
    def save_checkpoint(self):

        info = {}
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        checkpoint_path = os.path.join(self.config.OUTPUT, "sample_result.pth.tar")
        if is_main_process():
            torch.save(info, checkpoint_path)
            logger.info(f"save checkpoint to {checkpoint_path}")

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path)
        self.top_accuracies = info['top_accuracies']
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']

        logger.info(f"load checkpoint from {self.checkpoint_path}")
        return True
    
    
    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        
        self.model_without_ddp.set_sample_config(cand)

        n_parameters = self.model_without_ddp.flops()
        info['params'] = n_parameters
        
        if info['params'] > self.parameters_limits:
            logger.info('parameters limit exceed')
            return False

        if info['params'] < self.min_parameters_limits:
            logger.info('under minimum parameters limit')
            return False
        
        logger.info(f"cand:{cand}, params:{info['params']}")
        acc1, _, _ = validate(self.config, self.val_loader, self.model)
        info['acc'] = acc1
        logger.info(f"Evaluation Result (Top-1 Acc): {info['acc']}")        
        info['visited'] = True
        
        return True


    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand
                
    def get_random_cand(self):
        cand_tuple = self.rcg.random()

        return tuple(cand_tuple)
        
    
    def get_random(self, num=100):
        logger.info('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logger.info('random {}/{}'.format(len(self.candidates), num))
        logger.info('random_num = {}'.format(len(self.candidates)))

def main(args, config):
    _, _, _, data_loader_val, _= build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    
    model.set_sampler_info(config.NAS.SEARCH_SPACE)
    logger.info("Register search space")
    logger.info(f"*FLOPs of largest subnetworks: {model.max_flops}")
    logger.info(f"*FLOPs of smallest subnetworks: {model.min_flops}")
    
    model.cuda()

    if args.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    logger.info(str(model))

    model = torch.nn.parallel.DistributedDataParallel(
       model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    # Set the largest by default
    if config.NAS.INIT_CONFIG is None:
        model_without_ddp.set_largest_config()
    else:
        model_without_ddp.set_sample_config(config.NAS.INIT_CONFIG)
    
    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)

    t = time.time()
    
    searcher = SubnetSampler(args, model, model_without_ddp, config.NAS.SEARCH_SPACE, data_loader_val, config)

    searcher.get_random(num=args.sample_num)
    searcher.save_checkpoint()

    print('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))

@torch.no_grad()
def validate(config, data_loader, model, num_classes=1000):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
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
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(config.SEED) #This was set in RandomCandGenerator
    cudnn.benchmark = True


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
            wandb.init(project="sparsity", config=config_dict,
                       entity='max410011',
                       dir=wandb_output_path)

    main(args, config)