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


def parse_option():
    parser = argparse.ArgumentParser(
        'TinyViT training and evaluation script', add_help=False)
    parser.add_argument("--sparse-weights-path", type=str, required=True)
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config(args)
    return args, config



if __name__ == '__main__':
    args, config = parse_option()
    model = build_model(config)
    
    
    if config.MODEL.PRETRAINED:
        ckpt = torch.load(config.MODEL.PRETRAINED, map_location = 'cpu')
        print("Finish loading model")
        print(model.load_state_dict(ckpt['model'], strict = False))
    else:
        raise ValueError("Please provide the pretrained weight")
    

    print("Setting sparse configs to supernet")
    print("*Config:", config.NAS.INIT_CONFIG)
    
    model.set_real_sparse_weight(config.NAS.INIT_CONFIG)
    torch.save({"model": model.state_dict()}, args.sparse_weights_path)
    print(f"Save sparse weights to {args.sparse_weights_path}")
    
    
    
    
    
    
    
