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

from nas_utils import LinearEpsilonScheduler, CandidatePool
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
    
    model.set_sampler_info(config.NAS.SEARCH_SPACE, True)
    logger.info("Register search space")
    logger.info(f"*FLOPs of largest subnetworks: {model.max_flops}")
    logger.info(f"*FLOPs of smallest subnetworks: {model.min_flops}")
    
    if not args.only_cpu:
        model.cuda()

    if args.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = build_optimizer(config, model)

    if not args.only_cpu:
        model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters = False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
        
    # Set the largest by default
    if config.NAS.INIT_CONFIG is None:
        model_without_ddp.set_largest_config()
    else:
        model_without_ddp.set_sample_config(config.NAS.INIT_CONFIG)
    
    logger.info(str(model_without_ddp))
    
    loss_scaler = NativeScalerWithGradNormCount(grad_scaler_enabled=config.AMP_ENABLE)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops}")

    lr_scheduler = build_scheduler(config, optimizer, len(
        data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    
    epsilon3_scheduler = LinearEpsilonScheduler(total_epochs = config.TRAIN.EPOCHS,
                                            min_eps = config.NAS.SAMPLE_POLICY.EPS3.MIN,
                                            max_eps = config.NAS.SAMPLE_POLICY.EPS3.MAX,
                                            patient_epochs = config.NAS.SAMPLE_POLICY.EPS3.PATIENT_EPOCHS,
                                            fixed_epochs = config.NAS.SAMPLE_POLICY.EPS3.FIX_EPOCHS)
    
    epsilon4_scheduler = LinearEpsilonScheduler(total_epochs = config.TRAIN.EPOCHS,
                                            min_eps = config.NAS.SAMPLE_POLICY.EPS4.MIN,
                                            max_eps = config.NAS.SAMPLE_POLICY.EPS4.MAX,
                                            patient_epochs = config.NAS.SAMPLE_POLICY.EPS4.PATIENT_EPOCHS,
                                            fixed_epochs = config.NAS.SAMPLE_POLICY.EPS4.FIX_EPOCHS)
    
    filter_threshold = 0 # the choice filtering threshold

    cand_pool = CandidatePool(candidate_pool_size = config.NAS.SAMPLE_POLICY.CAND_POOL_SIZE)
    
    if config.DISTILL.ENABLED:
        # we disable MIXUP and CUTMIX when knowledge distillation
        assert len(
            config.DISTILL.TEACHER_LOGITS_PATH) > 0, "Please fill in DISTILL.TEACHER_LOGITS_PATH"
        criterion = SoftTargetCrossEntropy()
    else:
        if config.AUG.MIXUP > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif config.MODEL.LABEL_SMOOTHING > 0.:
            criterion = LabelSmoothingCrossEntropy(
                smoothing=config.MODEL.LABEL_SMOOTHING)
        else:
            criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger, candidate_pool=cand_pool)
        model_without_ddp.set_sample_config(config.NAS.INIT_CONFIG)
        acc1, acc5, loss = validate(args, config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(args, config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return
    
    if config.NAS.PROXY.ENABLE:
        data_loader_proxy = build_proxy_loader(config)
        proxy_samples_lists = []
        proxy_target_lists = []
        
        for samples, targets in data_loader_proxy:
            proxy_samples_lists.append(samples)
            proxy_target_lists.append(targets)
        
        proxy_samples = torch.cat(proxy_samples_lists, dim = 0).cuda(non_blocking = True)
        proxy_targets = torch.cat(proxy_target_lists, dim = 0).cuda(non_blocking = True)
        print(
            f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build proxy tensor with shape:{proxy_samples.shape}")
    else:
        proxy_samples = None
        proxy_targets = None    
    
    nas_test_config_list = []
    if config.NAS.TEST_CONFIG.TEST_SUBNET:
        nas_test_config_list += ['']
    if config.NAS.TEST_CONFIG.UNIFORM_SUBNETS:
        nas_test_config_list += config.NAS.TEST_CONFIG.UNIFORM_SUBNETS

    max_accuracy_list = [0.0 for _ in range(len(nas_test_config_list))]


    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # set_epoch for dataset_train when distillation
        if hasattr(dataset_train, 'set_epoch'):
            dataset_train.set_epoch(epoch)
        data_loader_train.sampler.set_epoch(epoch)

        if (epoch % config.NAS.SAMPLE_POLICY.FREQ_SAMPLE_EPOCHS == 0):# and (epoch > 0):
            filter_threshold = update_choices_weights(config, model, data_loader_proxy, current_epoch=epoch,
                                   eps5 = filter_threshold, eps5_v_num = config.NAS.SAMPLE_POLICY.NUM_CONV_VISIT)
            logger.info(f"The probabilities of sparsity choices in the {epoch}-th epoch:  {model.module.get_sampler_info()}")

        if config.DISTILL.ENABLED:
            train_acc1, train_acc5, train_loss = train_one_epoch_distill_using_saved_logits(
                args, config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, 
                loss_scaler, 
                # nas related parameter
                eps3 = epsilon3_scheduler.get_epsilon(epoch), eps4 = epsilon4_scheduler.get_epsilon(epoch),
                eps5 = filter_threshold, eps5_v_num = config.NAS.SAMPLE_POLICY.NUM_CONV_VISIT, 
                cand_pool = cand_pool, proxy_samples = proxy_samples, proxy_targets = proxy_targets)
        else:
            train_acc1, train_acc5, train_loss = train_one_epoch(args, config, model, criterion,
                            data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, 
                            # nas related parameter
                            eps3 = epsilon3_scheduler.get_epsilon(epoch), eps4 = epsilon4_scheduler.get_epsilon(epoch), 
                            eps5 = filter_threshold, eps5_v_num = config.NAS.SAMPLE_POLICY.NUM_CONV_VISIT,
                            cand_pool = cand_pool, proxy_samples = proxy_samples, proxy_targets = proxy_targets
                            )
            
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp,
                            max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, candidate_pool=cand_pool)

        log_stats = {}
        for i, nas_test_config in enumerate(nas_test_config_list):
            if nas_test_config:
                test_config = [nas_test_config for _ in range(config.NAS.NUM_CHOICES_BLOCKS)]
                model_without_ddp.set_sample_config(test_config)
            elif config.NAS.TEST_CONFIG.TEST_SUBNET: # TEST_SUBNET for [''] config
                model_without_ddp.set_sample_config(config.NAS.TEST_CONFIG.TEST_SUBNET)

            logger.info(
                f"FLOPS of the subnetwork: {model_without_ddp.flops()}")

            acc1, acc5, loss = validate(args, config, data_loader_val, model)
            logger.info(
                f"Accuracy of the {nas_test_config} subnetwork on the {len(dataset_val)} test images: {acc1:.1f}%")
            max_accuracy_list[i] = max(max_accuracy_list[i], acc1)
            logger.info(f'{nas_test_config} Max accuracy: {max_accuracy_list[i]:.2f}%')
            log_stats = {**log_stats, **{f"test/{nas_test_config}_acc1": acc1,
                                         f"test/{nas_test_config}_loss": loss,
                                         }}
        
        if is_main_process() and args.use_wandb:
            wandb.run.summary['epoch'] = epoch
            for i, nas_test_config in enumerate(nas_test_config_list):
                wandb.run.summary[f'best_{nas_test_config}_acc1'] = max_accuracy_list[i]
            wandb.log({**log_stats, **{'train/lr': optimizer.param_groups[0]['lr'], 
                                       'train/acc@1': train_acc1,
                                       'train/loss': train_loss
                                       }}, step=epoch)
        
        max_accuracy = max_accuracy_list[0] ### work around
            

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def is_valid_grad_norm(num):
    if num is None:
        return False
    return not bool(torch.isinf(num)) and not bool(torch.isnan(num))


def set_bn_state(config, model):
    if config.TRAIN.EVAL_BN_WHEN_TRAINING:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()

def update_choices_weights(config, model, data_loader_val, current_epoch, eps5 = None, eps5_v_num = None):
    '''
    return the threshold for the following random function
    '''
    num_subnets = config.NAS.SAMPLE_POLICY.NUM_SAMPLE_SUBNETS
    epsilon3_1_scheduler = LinearEpsilonScheduler(total_epochs = num_subnets,
                                            min_eps = config.NAS.SAMPLE_POLICY.EPS3.MIN,
                                            max_eps = config.NAS.SAMPLE_POLICY.EPS3.MAX,
                                            patient_epochs = num_subnets / 3,
                                            fixed_epochs = num_subnets / 5)
    subnets = []
    for subnet_i in range(num_subnets):
        eps3_1 = epsilon3_1_scheduler.get_epsilon(subnet_i)
        cfg = model.module.random_config_fn(weighted_mode = eps3_1, filter_ratio = 0.0, filter_threshold = eps5, confident_num_visit = eps5_v_num)
        subnets.append(cfg)

    acc_record = []
    for subnet in subnets:
        # set the configuration of subnetwork
        model.module.set_sample_config(subnet)
        acc1, acc5, loss = validate(args, config, data_loader_val, model)
        acc_record.append(acc1)
        logger.info(f'Config:{subnet}\n Acc1:{acc1} Acc5:{acc5} Loss:{loss}')
        model.module.update_sample_prob(subnet, acc1)
    
    acc_record.sort()
    avg_acc1 = sum(acc_record) / len(acc_record)
    acc_thres = 0
    if avg_acc1 > 20.0 and acc_record[-1] - avg_acc1 >= 20.0:
        acc_thres = avg_acc1 - 20.0
    elif acc_record[-1] - avg_acc1 >= 5.0:
        acc_thres = avg_acc1 - (acc_record[-1] - avg_acc1)
    else:
        acc_thres = avg_acc1 - 2.5
    if current_epoch > 0:
        model.module.update_filter(0, acc_thres, eps5_v_num)
    return acc_thres

def get_subnets(args, config, model, cand_pool:CandidatePool, eps3 = None, eps4 = None, eps5 = None, eps5_v_num = None, proxy_samples = None, proxy_targets = None):
    flops_of_largest_subnet =  model.module.flops(model.module.get_largest_config())
    flops_of_smallest_subnet = model.module.flops(model.module.get_smallest_config())
    flops_target = flops_of_largest_subnet * config.NAS.TARGET_COMP_RATIO
    
    num_subnets = config.NAS.SAMPLE_POLICY.NUM_SAMPLE_SUBNETS
    num_kept_subnets = config.NAS.SAMPLE_POLICY.NUM_KEPT_SUBNETS
    filter_policy = config.NAS.SAMPLE_POLICY.FILTERED_POLICY # how to filter the subnets
    enable_cand_pool = config.NAS.SAMPLE_POLICY.ENABLE_CAND_POOL
    
    subnets = []
    for _ in range(num_subnets):
        cfg = model.module.random_config_fn(weighted_mode = eps3, filter_ratio = eps4, filter_threshold = eps5, confident_num_visit = eps5_v_num)
        subnets.append(cfg)
    
    
    if filter_policy is None:
        return subnets
    elif filter_policy == 'greedy':
        candidate_subnets_with_scores_and_flops = []
        proxy_criterion = torch.nn.CrossEntropyLoss()
        for subnet in subnets:
            # set the configuration of subnetwork
            model.module.set_sample_config(subnet)
            cur_flop = model.module.flops()
            # inference
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                output = model(proxy_samples)
                proxy_loss = proxy_criterion(output, proxy_targets)
            proxy_loss = proxy_loss.item()
            # Synchronization and calculate the metric with respect to flops
            t = torch.tensor([proxy_loss], device = 'cuda')
            dist.barrier()
            dist.all_reduce(t)
            proxy_losses  = t.tolist()[0]
            
            
            candidate_subnets_with_scores_and_flops.append((subnet, proxy_losses, cur_flop))
        
        candidate_subnets_with_scores_and_flops = sorted(candidate_subnets_with_scores_and_flops, key=lambda x: x[1])
        candidate_subnets_with_scores_and_flops = candidate_subnets_with_scores_and_flops[:min(len(candidate_subnets_with_scores_and_flops), num_kept_subnets)]

        if enable_cand_pool:
            for subnet, score, flops in candidate_subnets_with_scores_and_flops:
                if flops <= flops_target:
                    cand_pool.add_one_subnet_with_score_and_flops(subnet, score, flops)
        return [subnet for subnet, _, _, in candidate_subnets_with_scores_and_flops]
    else:
        raise NotImplementedError(f"filter policy:{filter_policy} is not supported")


def train_one_epoch(args, config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler,
                    eps3 = None, eps4 = None, eps5 = None, eps5_v_num = None, cand_pool = None, proxy_samples = None, proxy_targets = None):
    model.train()
    set_bn_state(config, model)
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    start = time.time()
    end = time.time()
    
    filtered_subnets_pools = get_subnets(args, config, model, cand_pool, eps3, eps4, eps5, eps5_v_num, proxy_samples, proxy_targets)
    filtered_subnets_pools_update_freq = config.NAS.SAMPLE_POLICY.NUM_KEPT_SUBNETS
    logger.info(f"=> Current eps3: {eps3}, eps4: {eps4}, eps5: {eps5}")
    for idx, (samples, targets) in enumerate(data_loader):
        normal_global_idx = epoch * NORM_ITER_LEN + \
            (idx * NORM_ITER_LEN // num_steps)

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            original_targets = targets.argmax(dim=1)
        else:
            original_targets = targets

        # resample k paths using path filtering algorithm every k iterations
        if ((idx) % filtered_subnets_pools_update_freq) == 0:
            filtered_subnets_pools = get_subnets(args, config, model, cand_pool, eps3, eps4, eps5, eps5_v_num, proxy_samples, proxy_targets)
        
        # set the configuration of subnetworks
        subnet = filtered_subnets_pools[(idx) % filtered_subnets_pools_update_freq]
        model.module.set_sample_config(subnet)
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        with torch.no_grad():
            acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))
        acc1_meter.update(acc1.item(), targets.size(0))
        acc5_meter.update(acc5.item(), targets.size(0))

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if is_valid_grad_norm(grad_norm):
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB\t'
                f'eps3 {eps3:.4f}\t'
                f'eps4 {eps4:.4f}\t'
                f'eps5 {eps5:.4f}'
                )

            # if is_main_process() and args.use_wandb:
            #     wandb.log({
            #         "train/acc@1": acc1_meter.val,
            #         # "train/acc@5": acc5_meter.val,
            #         "train/loss": loss_meter.val,
            #         "train/grad_norm": norm_meter.val,
            #         "train/loss_scale": scaler_meter.val,
            #         "train/lr": lr,
            #     }, step=normal_global_idx)

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

def train_one_epoch_distill_using_saved_logits(args, config, model, criterion, data_loader, 
                                               optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler,
                                               eps3 = None, eps4 = None, eps5 = None, eps5_v_num = None, 
                                               cand_pool = None, proxy_samples = None, proxy_targets = None):
    model.train()
    set_bn_state(config, model)
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()
    data_tic = time.time()

    num_classes = config.MODEL.NUM_CLASSES
    topk = config.DISTILL.LOGITS_TOPK
    
    filtered_subnets_pools = get_subnets(args, config, model, cand_pool, eps3, eps4, eps5, eps5_v_num, proxy_samples, proxy_targets)
    filtered_subnets_pools_update_freq = config.NAS.SAMPLE_POLICY.NUM_KEPT_SUBNETS
        
    for idx, ((samples, targets), (logits_index, logits_value, seeds)) in enumerate(data_loader):
        
        normal_global_idx = epoch * NORM_ITER_LEN + \
            (idx * NORM_ITER_LEN // num_steps)

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets, seeds)
            original_targets = targets.argmax(dim=1)
        else:
            original_targets = targets
        meters['data_time'].update(time.time() - data_tic)

        # resample k paths using path filtering algorithm every k iterations
        if ((idx) % filtered_subnets_pools_update_freq) == 0:
            filtered_subnets_pools = get_subnets(args, config, model, cand_pool, eps3, eps4, eps5, eps5_v_num, proxy_samples, proxy_targets)
        
        subnet = filtered_subnets_pools[(idx) % filtered_subnets_pools_update_freq]
        logger.info(f'Subnet config: {subnet}')
        # set the configuration of subnetworks
        model.module.set_sample_config(subnet)
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        # recover teacher logits
        logits_index = logits_index.long()
        logits_value = logits_value.float()
        logits_index = logits_index.cuda(non_blocking=True)
        logits_value = logits_value.cuda(non_blocking=True)
        minor_value = (1.0 - logits_value.sum(-1, keepdim=True)
                       ) / (num_classes - topk)
        minor_value = minor_value.repeat_interleave(num_classes, dim=-1)
        outputs_teacher = minor_value.scatter_(-1, logits_index, logits_value)

        loss = criterion(outputs, outputs_teacher)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        # compute accuracy
        real_batch_size = len(original_targets)
        acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))
        meters['train_acc1'].update(acc1.item(), real_batch_size)
        meters['train_acc5'].update(acc5.item(), real_batch_size)
        teacher_acc1, teacher_acc5 = accuracy(
            outputs_teacher, original_targets, topk=(1, 5))
        meters['teacher_acc1'].update(teacher_acc1.item(), real_batch_size)
        meters['teacher_acc5'].update(teacher_acc5.item(), real_batch_size)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), real_batch_size)
        if is_valid_grad_norm(grad_norm):
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()
        data_tic = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)

            extra_meters_str = ''
            for k, v in meters.items():
                extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'{extra_meters_str}'
                f'mem {memory_used:.0f}MB\t'
                f'eps3 {eps3:.4f}\t'
                f'eps4 {eps4:.4f}\t'
                f'eps5 {eps5:.4f}\t'
                f'cur_pool_size {cand_pool.get_size()}')

            # if is_main_process() and args.use_wandb:
            #     acc1_meter, acc5_meter = meters['train_acc1'], meters['train_acc5']
            #     wandb.log({
            #         "train/acc@1": acc1_meter.val,
            #         # "train/acc@5": acc5_meter.val,
            #         "train/loss": loss_meter.val,
            #         "train/grad_norm": norm_meter.val,
            #         "train/loss_scale": scaler_meter.val,
            #         "train/lr": lr,
            #     }, step=normal_global_idx)
    epoch_time = time.time() - start
    extra_meters_str = f'Train-Summary: [{epoch}/{config.TRAIN.EPOCHS}]\t'
    for k, v in meters.items():
        v.sync()
        extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
    logger.info(extra_meters_str)
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return meters['train_acc1'].avg, meters['train_acc5'].avg, loss_meter.avg


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


@torch.no_grad()
def throughput(data_loader, model, logger):
    # we follow the throughput measurement of LeViT repo (https://github.com/facebookresearch/LeViT/blob/main/speed_test.py)
    model.eval()

    T0, T1 = 10, 60
    images, _ = next(iter(data_loader))
    batch_size, _, H, W = images.shape
    inputs = torch.randn(batch_size, 3, H, W).cuda(non_blocking=True)

    # trace model to avoid python overhead
    model = torch.jit.trace(model, inputs)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start = time.time()
    with torch.cuda.amp.autocast():
        while time.time() - start < T0:
            model(inputs)
    timing = []
    torch.cuda.synchronize()
    with torch.cuda.amp.autocast():
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    throughput = batch_size / timing.mean().item()
    logger.info(f"batch_size {batch_size} throughput {throughput}")


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
