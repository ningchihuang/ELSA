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


from yacs.config import CfgNode as CN
from contextlib import redirect_stdout

def parse_option():
    parser = argparse.ArgumentParser(
        'TinyViT training and evaluation script', add_help=False)
    add_common_args(parser)
    # ea_searching_results
    parser.add_argument("--ea-result-path", type=str, help='path to the evolutionary search results (.tar)')
    parser.add_argument("--subnet-cfg-name", type=str, 
                        help='name of the subnet config to be dumped. If not specified, \
                        we will use the name of supernet with the flops and acc as suffic for default')
    args = parser.parse_args()

    config = get_config(args)

    return args, config

def get_config_with_highest_accuracy_from_ea_result(path):
    def parse_ckpt(m):
        X1 = []
        Y1 = []
        Z1 = []
        for k, v in m['vis_dict'].items():
            if 'params' not in v:
                continue
            if 'acc' not in v:
                continue
            X1.append(v['params'])
            Y1.append(v['acc'])
            Z1.append(k)
        
        return np.array(X1), np.array(Y1), np.array(Z1)
    
    ckpt = torch.load(path)
    flops, accs, cfgs = parse_ckpt(ckpt)
    max_acc_config_idx = np.argmax(accs)
    return flops[max_acc_config_idx], accs[max_acc_config_idx], list([list(cfg) for cfg in cfgs[max_acc_config_idx]])

def dump_to_yaml_file(config):
    new_config = CN()
    new_config.MODEL = CN()
    new_config.MODEL.TYPE = config.MODEL.TYPE
    new_config.MODEL.NAME = config.MODEL.NAME
    
    if config.MODEL.TYPE == 'sparse_swin':
        new_config.MODEL.SWIN = config.MODEL.SWIN.clone()
    elif config.MODEL.TYPE == 'sparse_deit':
        new_config.MODEL.DEIT = config.MODEL.DEIT.clone()
    
    with open(f'{config.MODEL.NAME}.yaml', 'a') as f:
        with redirect_stdout(f): print(new_config)
        f.write("\n")
    

def main(args, config):    
    config.defrost()
    
    flops, acc, sparse_config = get_config_with_highest_accuracy_from_ea_result(args.ea_result_path)
    print("FLOPs: ", flops)
    print("Acc:", acc)
    print("Searched sparse config:", sparse_config)
    
    new_config = CN()
    new_config.MODEL = CN()
    new_config.MODEL.TYPE = config.MODEL.TYPE
    new_config.MODEL.NAME = f"{config.MODEL.NAME}_{flops}G_{acc}"
    
    if config.MODEL.TYPE == 'sparse_swin':
        new_config.MODEL.SWIN = config.MODEL.SWIN.clone()
    elif config.MODEL.TYPE == 'sparse_deit':
        new_config.MODEL.DEIT = config.MODEL.DEIT.clone()
        
    new_config.NAS = config.NAS.clone()
    new_config.NAS.TEST_CONFIG.UNIFORM_SUBNETS = []
    new_config.NAS.TEST_CONFIG.TEST_SUBNET = sparse_config
    new_config.NAS.SEARCH_SPACE = [[cfg] for cfg in sparse_config]
    new_config.NAS.INIT_CONFIG = sparse_config
    
    output_file_name = args.subnet_cfg_name if args.subnet_cfg_name else f"{new_config.MODEL.NAME}.yaml"
    with open(f'{new_config.MODEL.NAME}.yaml', 'a') as f:
        with redirect_stdout(f): print(new_config)
        f.write("\n")
    

if __name__ == '__main__':
    args, config = parse_option()
    main(args, config)
