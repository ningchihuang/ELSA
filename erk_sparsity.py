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
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist


from config import get_config
from models import build_model

from utils import load_checkpoint, load_pretrained, save_checkpoint,\
    NativeScalerWithGradNormCount,\
    auto_resume_helper, is_main_process,\
    add_common_args,\
    get_git_info

from models.remap_layer import RemapLayer
remap_layer_22kto1k = RemapLayer('./imagenet_1kto22k.txt')
exclude_module_name = ['head']

def parse_option():
    parser = argparse.ArgumentParser(
        'TinyViT training and evaluation script', add_help=False)
    parser.add_argument('--target-sparsity', type=float, default=0.5, help='target sparsity level')
    parser.add_argument('--bases', type=int, default=4, help='M for N:M sparsity')

    add_common_args(parser)
    args = parser.parse_args()
    config = get_config(args)
    return args, config


def get_search_space(bases, num_blocks):
    cfg = []
    cand = 1
    while cand <= bases:
        cfg.append([cand, bases])
        cand *= 2
    return [cfg for _ in range(num_blocks)] 

def _is_prunable_module(m, name = None):
    if name is not None and name in exclude_module_name:
        return False
    return isinstance(m, nn.Linear)


def get_modules(model):
    modules = []
    for name, m in model.named_modules():
        if _is_prunable_module(m, name):
            modules.append(m)
    return modules

def get_weights(model):
    weights = []
    for name, m in model.named_modules():
        if _is_prunable_module(m, name):
            weights.append(m.weight)
    return weights

def _amounts_from_eps(unmaskeds,ers,amount):

    num_layers = ers.size(0)
    layers_to_keep_dense = torch.zeros(num_layers)
    total_to_survive = (1.0-amount)*unmaskeds.sum() # Total to keep.

    # Determine some layers to keep dense.
    is_eps_invalid = True
    while is_eps_invalid:
        unmasked_among_prunables = (unmaskeds*(1-layers_to_keep_dense)).sum()
        to_survive_among_prunables = total_to_survive - (layers_to_keep_dense*unmaskeds).sum()

        
        ers_of_prunables = ers*(1.0-layers_to_keep_dense)
        survs_of_prunables = torch.round(to_survive_among_prunables*ers_of_prunables/ers_of_prunables.sum())

        layer_to_make_dense = -1
        max_ratio = 1.0
        
        for idx in range(num_layers):
            if layers_to_keep_dense[idx] == 0:
                if survs_of_prunables[idx]/unmaskeds[idx] > max_ratio:
                    layer_to_make_dense = idx
                    max_ratio = survs_of_prunables[idx]/unmaskeds[idx]

        if layer_to_make_dense == -1:
            is_eps_invalid = False
        else:
            layers_to_keep_dense[layer_to_make_dense] = 1

    amounts = torch.zeros(num_layers)

    for idx in range(num_layers):
        if layers_to_keep_dense[idx] == 1:
            amounts[idx] = 0.0
        else:
            amounts[idx] = 1.0 - (survs_of_prunables[idx]/unmaskeds[idx])
    return amounts

def _compute_erks(model):
    wlist = get_weights(model)
    erks = torch.zeros(len(wlist))
    for idx,w in enumerate(wlist):
        if w.dim() == 4:
            erks[idx] = w.size(0)+w.size(1)+w.size(2)+w.size(3)
        else:
            erks[idx] = w.size(0)+w.size(1)
    return erks


def _count_unmasked_weights(model):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    """
    mlist = get_modules(model)
    unmaskeds = []
    for m in mlist:
        unmaskeds.append(m.weight.size(0) * m.weight.size(1))
    return torch.FloatTensor(unmaskeds)


def get_n_m_sparsity(model,amount, bases = 4):
    unmaskeds = _count_unmasked_weights(model)
    erks = _compute_erks(model)
    amounts = _amounts_from_eps(unmaskeds,erks,amount)

    print("*Unstructured sparsity level decided by ERK", amounts)
    
    def find_nearest_fraction_index(fractions, x):
        min_diff = float('inf')  # initialize with infinity
        nearest_index = None  # initialize with None
        
        for i, (numerator, denominator) in enumerate(fractions):
            # Calculate the absolute difference between x and the fraction
            diff = abs(x - numerator/denominator)
            
            # Update the min_diff and nearest_index if a smaller difference is found
            if diff < min_diff:
                min_diff = diff
                nearest_index = i
        return nearest_index
    
    
    search_space = get_search_space(bases=4, num_blocks=erks.size(0))
    
    cfgs = []
    for choices, amount in zip(search_space, amounts):
        c = choices[find_nearest_fraction_index(choices, 1 - amount.item())]
        cfgs.append(c)

    return cfgs
    

if __name__ == '__main__':
    args, config = parse_option()
    model = build_model(config)
    sparsity_config = get_n_m_sparsity(model, args.target_sparsity)
    print("*N:M sparsity configs:", sparsity_config)
    print([[cfg] for cfg in sparsity_config])
    
    
    
    
    
    
    
