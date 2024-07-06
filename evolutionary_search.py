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

from timm.utils import accuracy
from my_meter import AverageMeter

from config import get_config
from models import build_model
from data import build_loader, build_proxy_loader
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint,\
    auto_resume_helper, is_main_process,\
    get_git_info, add_common_args
from models.remap_layer import RemapLayer
remap_layer_22kto1k = RemapLayer('./imagenet_1kto22k.txt')

from pathlib import Path
import matplotlib.pyplot as plt
from nas_utils import AdaptiveCandGenerator, LinearEpsilonScheduler

def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script', add_help=False)
    add_common_args(parser)
    # evolution search parameters
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--s_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--param-limits', type=float, default=5.6)
    parser.add_argument('--min-param-limits', type=float, default=5)
    parser.add_argument('--acc-min-lim', type=float, default=70)
    parser.add_argument('--acc-max-lim', type=float, default=80)

    parser.add_argument('--eval-pareto-coef', default=1.0, type=float, help="coef for Pareto when evaluating")
    parser.add_argument('--eval-acc-coef', default=1.0, type=float, help="coef for accuracy when evaluating")
    parser.add_argument('--eval-acc-threshold', default=70.0, type=float, help="threshold for accuracy when evaluating")
    parser.add_argument('--eval-param-coef', default=1.0, type=float, help="coef for parameters when evaluating")

    parser.add_argument('--watch-checkpoints', metavar='FILE', type=Path, nargs='+',
                        help='one or more .pth.tar files to combine, and then output a scatter chart')
    parser.add_argument('--debug', default=False, type=bool, help='debug_mode: use fake acc1 to reduce debug time')

    args = parser.parse_args()

    config = get_config(args)

    return args, config

class EvolutionSearcher():
    def __init__(self, args, search_space, model, model_without_ddp, val_loader, config):
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        self.acc_min_lim = args.acc_min_lim
        self.acc_max_lim = args.acc_max_lim
        self.val_loader = val_loader
        self.s_prob =args.s_prob
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], self.population_num: [], 'pareto': []}
        self.epoch = 0
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.search_space = search_space
        self.config_length = len(search_space) # num of choice blocks, e.g., [[1, 4], [2, 4], [4, 4]]
        self.acg = AdaptiveCandGenerator(self.search_space, False)
        tmp_sparsity_config = []
        for block_i in self.search_space:
            tmp_block = []
            for config_i in block_i:
                tmp_block.append(tuple(config_i))
            tmp_sparsity_config.append(tmp_block)
        self.search_space = tmp_sparsity_config
        self.config = config

        self.pareto_coef = args.eval_pareto_coef
        self.acc_coef = args.eval_acc_coef
        self.params_coef = args.eval_param_coef
        self.dense_params_num = self.model_without_ddp.num_params() / 1e6
        dense_config = []
        for configs in self.search_space:
            dense_config.append(configs[-1]) 
        self.dense_acc = self.get_acc1(tuple(dense_config))
        smallest_config = []
        for configs in self.search_space:
            smallest_config.append(configs[0])
        self.sparsest_acc = self.get_acc1(tuple(smallest_config))
        self.acc_threshold = max(self.sparsest_acc, args.eval_acc_threshold)
        self.smallest_params_num = self.model_without_ddp.num_params() / 1e6
        for config in self.search_space[0]:
            tmp_cand = []
            for _ in range(self.config_length):
                tmp_cand.append(config)
            self.acg.update_adapt_fitness(tuple(tmp_cand), self.get_acc1(tuple(tmp_cand)))
        self.debug_mode = args.debug
        self.bin_num = int(min(self.config_length, self.config_length / len(self.search_space[0]) * ((self.parameters_limits - self.min_parameters_limits) / self.smallest_params_num)))

        logger.info(f"Create evolution searcher with search space:{self.search_space}")

    def save_checkpoint(self):
        info = {}
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.config.OUTPUT, "checkpoint-{}.pth.tar".format(self.epoch))
        if is_main_process():
            torch.save(info, checkpoint_path)
            logger.info(f'save checkpoint to: {checkpoint_path}')
            self.watch_checkpoint([checkpoint_path])

    def watch_checkpoint(self, checkpoint_paths):
        plt.rc('font', size=18)
        # plt.rc('figure', titlesize=14)
        plt.figure(figsize=(8.89,6.35))
        plt.xlabel('FLOPs (G)', fontsize=20)
        plt.xlim([self.min_parameters_limits, self.parameters_limits])
        plt.ylabel('ImageNet Top-1 Accuracy (%)', fontsize=20)
        plt.ylim([self.acc_min_lim, self.acc_max_lim])

        plt_x = np.empty((0,))
        plt_y = np.empty((0,))
        str_png_id = ''
        print(self.keep_top_k.keys())
        for checkpoint_path in checkpoint_paths:
            # Load the checkpoint file
            info = torch.load(checkpoint_path)
            print('load checkpoint from ', checkpoint_path)
            str_png_id_i = str(checkpoint_path).split("-", -1)[-1].split(".", 1)[0]
            str_png_id += "_" + str_png_id_i

            tmp_x = []
            tmp_y = []
            for i, cand in enumerate(info['keep_top_k'][self.population_num]):
                print('No.{} {} \nTop-1 val acc = {}, params = {}'.format(
                    i + 1, cand, info['vis_dict'][cand]['acc'], info['vis_dict'][cand]['params']))
                tmp_x.append(info['vis_dict'][cand]['params'])
                tmp_y.append(info['vis_dict'][cand]['acc'])
                plt_x = np.append(plt_x, tmp_x, axis=0)
                plt_y = np.append(plt_y, tmp_y, axis=0)
            plt.scatter(tmp_x, tmp_y)
        
        data = sorted(zip(plt_x, plt_y), key=lambda d: d[0])
        frontier_x = []
        frontier_y = []
        max_y = 0
        for xi, yi in data:
            if yi > max_y:
                frontier_x.append(xi)
                frontier_y.append(yi)
                max_y = yi

        # plt.scatter(plt_x, plt_y)
        plt.plot(frontier_x, frontier_y, 'r', linewidth=2)
        combined = [(xi, yi) for xi, yi in zip(frontier_x, frontier_y)]
        print(combined)
        with open('{}/EA_pareto_{}_{}_iter{}.txt'.format(self.config.OUTPUT, self.min_parameters_limits, self.parameters_limits, str_png_id), 'w', encoding="utf-8") as f:
            for cand in info['keep_top_k']['pareto']:
                print(cand, info['vis_dict'][cand]['params'], info['vis_dict'][cand]['acc'], info['vis_dict'][cand]['v_by'], file=f)
        f.closed
        with open('{}/EA_population_{}_{}_iter{}.txt'.format(self.config.OUTPUT, self.min_parameters_limits, self.parameters_limits, str_png_id), 'w', encoding="utf-8") as f:
            for cand in info['keep_top_k'][self.population_num]:
                print(cand, info['vis_dict'][cand]['params'], info['vis_dict'][cand]['acc'], info['vis_dict'][cand]['v_by'], file=f)
        f.closed
        plt.savefig('{}/EA_scatter_{}_{}_iter{}.png'.format(self.config.OUTPUT, self.min_parameters_limits, self.parameters_limits, str_png_id))
    
    def get_acc1(self, cand):
        assert isinstance(cand, tuple)
        self.model_without_ddp.set_sample_config(cand)
        n_parameters = self.model_without_ddp.flops()
        acc1, _, _ = validate(self.config, self.val_loader, self.model)
        print(cand, n_parameters, acc1)

        return acc1

    def is_legal(self, cand, visited_by):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            info['v_by'].append(visited_by)
            return False
        info['visited'] = True
        info['v_by'] = [visited_by]
        
        self.model_without_ddp.set_sample_config(cand)
        # print(cand)
        n_parameters = self.model_without_ddp.flops()
        info['params'] = n_parameters # sparsity level
        # print(n_parameters)
        logger.info(f'Cand: {cand} FLOPs: {n_parameters}')
        
        if info['params'] > self.parameters_limits:
            logger.info('parameters limit exceed')
            if self.config.NAS.PROXY.ENABLE:
                acc1, _, _ = validate(self.config, self.val_loader, self.model)
                self.acg.update_adapt_fitness(cand, acc1)
                self.acg.update_filter(0, self.acc_threshold, config.NAS.SAMPLE_POLICY.NUM_CONV_VISIT)
            return False

        if info['params'] < self.min_parameters_limits:
            logger.info('under minimum parameters limit')
            if self.config.NAS.PROXY.ENABLE:
                acc1, _, _ = validate(self.config, self.val_loader, self.model)
                self.acg.update_adapt_fitness(cand, acc1)
                self.acg.update_filter(0, self.acc_threshold, config.NAS.SAMPLE_POLICY.NUM_CONV_VISIT)
            return False
        
        logger.info(f"cand:{cand}, params:{info['params']}")
        if (self.debug_mode):
            info['acc'] = random.normalvariate(mu=(self.acc_max_lim+self.acc_max_lim)/2.0, sigma=(self.acc_max_lim-self.acc_max_lim)/6.0)
            print('Top-1 acc:', info['acc'], '(fake)')
        else:
            acc1, _, _ = validate(self.config, self.val_loader, self.model)
            info['acc'] = acc1

        logger.info(f"Evaluation Result (Top-1 Acc): {info['acc']}")
        self.acg.update_adapt_fitness(cand, acc1)
        self.acg.update_filter(0, self.acc_threshold, config.NAS.SAMPLE_POLICY.NUM_CONV_VISIT)
        
        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def update_pareto(self, candidates): # subnets has less params with larger acc values are better
        assert 'pareto' in self.keep_top_k
        print('select Pareto......')
        t = self.keep_top_k['pareto']
        t += candidates

        t.sort(key=lambda x: self.vis_dict[x]['params'], reverse=False)
        pareto_points = []
        max_y = self.acc_min_lim
        max_y_x = 0.0
        for point in t:
            if self.vis_dict[point]['acc'] > max_y:
                pareto_points.append(point)
                max_y = self.vis_dict[point]['acc']
                max_y_x = self.vis_dict[point]['params']
            elif (self.vis_dict[point]['params'] == max_y_x) and (self.vis_dict[point]['acc'] == max_y):
                pareto_points.append(point)
                max_y = self.vis_dict[point]['acc']
        self.keep_top_k['pareto'] = pareto_points # data type shold be modified
        smallest_pareto_point = self.keep_top_k['pareto'][0]
        self.acc_threshold = self.vis_dict[smallest_pareto_point]['acc']
        print('Pareto points have been updated!')
                
    def get_random_cand(self, weighted_mode = 0.0):#, filter_threshold = 0.1, confident_num_visit = 5):
        cand_tuple = self.acg.adaptiv_random(weighted_mode)#, 0.0, filter_threshold, confident_num_visit)
        return tuple(cand_tuple)
    
    def get_random(self, num):
        logger.info('random select ........')
        epsilon3_scheduler = LinearEpsilonScheduler(total_epochs = num,
                                            min_eps = self.config.NAS.SAMPLE_POLICY.EPS3.MIN,
                                            max_eps = self.config.NAS.SAMPLE_POLICY.EPS3.MAX,
                                            patient_epochs = self.config.NAS.SAMPLE_POLICY.EPS3.PATIENT_EPOCHS,
                                            fixed_epochs = self.config.NAS.SAMPLE_POLICY.EPS3.FIX_EPOCHS)

        cand_i = 0
        cand_j = 0
        while len(self.candidates) < num:
            eps3 = epsilon3_scheduler.get_epsilon(cand_i)
            cand_cfg = self.get_random_cand(eps3)
            if not self.is_legal(cand_cfg, 'rand'):
                if cand_j >= 5:
                    cand_j = 0
                    cand_i += 1
                else:
                    cand_j += 1
                continue
            self.candidates.append(cand_cfg)

            cand_j = 0
            cand_i += 1
            if cand_i >= num:
                cand_i = int(num / 2)
            logger.info('random {}/{}'.format(len(self.candidates), num))
        logger.info('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, s_prob, it_idx):
        assert k in self.keep_top_k
        logger.info('mutation ......')
        res = []
        # iter = 0
        max_iters = mutation_num * 10
        
        bin_num = self.bin_num
        bins = np.linspace(self.min_parameters_limits, self.parameters_limits, bin_num+1) # a sequence with 11 elements (10 intervals)
        bins_indices = np.digitize([self.vis_dict[key]['params'] for key in self.keep_top_k[k]], bins[1:]) # digitize does not need the leftmost bound
        bins_element_count = np.array([np.count_nonzero(bins_indices == b_idx) for b_idx in range(bin_num)])
        def bin_rand(bin_idx_i):
            muta_cand = []
            cand_bin_idx = bin_idx_i
            while cand_bin_idx == bin_idx_i:
                muta_cand = random.choice(self.keep_top_k[k])
                cand_bin_idx = np.digitize(self.vis_dict[muta_cand]['params'], bins[1:])
                print("cand: ", cand_bin_idx, "; bin: ", bin_idx_i)
            
            def cmp_choices(c1, c2): # c1>c2: 1, c1==c2: 0, c1<c2: -1
                x1, y1 = c1
                x2, y2 = c2
                ans = (x1 * y2) - (x2 * y1)
                if ans > 0:
                    return 1
                elif ans < 0:
                    return -1
                else: # ans == 0
                    return 0

            def points_mutation(muta_cand, muta_direction, pos_num):
                cand_gene_pos4muta = [] # a mask of possible points got mutation
                for i, c in enumerate(self.search_space):
                    has_muta_choice = False
                    for choice in list(c):
                        if cmp_choices(choice, muta_cand[i]) == muta_direction:
                            cand_gene_pos4muta.append(i)
                            break
                muta_pos_list = np.random.choice(cand_gene_pos4muta, size=pos_num, replace=False)
                for idx in range(self.config_length):
                    if idx in muta_pos_list:
                        possible_choices = []
                        for choice in self.search_space[idx]:
                            if cmp_choices(choice, muta_cand[idx]) == muta_direction:
                                possible_choices.append(choice)
                        muta_cand[idx] = tuple(random.choice(possible_choices))
                return tuple(muta_cand)

            if cand_bin_idx < bin_idx_i: # muta with larger choices
                return points_mutation(list(muta_cand), 1, abs(cand_bin_idx - bin_idx_i))
            else: # cand_bin_idx > bin_idx_i: muta with smaller choices
                return points_mutation(list(muta_cand), -1, abs(cand_bin_idx - bin_idx_i))
        def stack_bin_cand(random_func, bin_idx, batchsize=10):
            while True:
                cands = [random_func(bin_idx) for _ in range(batchsize)]
                for cand in cands:
                    print(cands)
                    if cand not in self.vis_dict:
                        self.vis_dict[cand] = {}
                    # info = self.vis_dict[cand]
                for cand in cands:
                    yield cand
        bin_less_element = []
        print(k / (bin_num + 1.0))
        print(bins_element_count)
        for bin_idx in range(bin_num):
            if bins_element_count[bin_idx] < k / (bin_num + 1.0):
                for bin_idx_j in range(bin_num):
                    if (abs(bin_idx_j - bin_idx) < bin_num / 2) and bins_element_count[bin_idx_j] > 0:
                        bin_less_element.append(bin_idx)
                        break
        print(bin_less_element)
        while len(res) < int(mutation_num / 2) and len(bin_less_element) > 0:
            bin_idx_less = random.choice(bin_less_element)
            cand_iter = stack_bin_cand(bin_rand, bin_idx_less, batchsize=5)
            cand_count = 0
            while len(res) < int(mutation_num / 2) and max_iters > 0 and cand_count < 5:
                max_iters -= 1
                cand = next(cand_iter)
                if not self.is_legal(cand, 'muta_bin' + it_idx):
                    continue
                res.append(cand)
                cand_count += 1
                logger.info('mutation_bin {}/{}'.format(len(res), mutation_num))

        logger.info('bin mutation_num = {}'.format(len(res)))

        gene_prob = np.array(list(self.acg.get_adapt_prob().values()))
        gene_prob_inv = (1.0 - gene_prob) / np.sum(1.0 - gene_prob, axis=0) # may have a bug if len(search_space[i])!=len(sparsity_config[j])

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))

            # sparsity ratio
            for idx in range(self.config_length):
                random_s = random.random()
                random_inv = np.random.choice([0, 1], p=[1.0 - s_prob, s_prob])
                tmp_gene_prob = gene_prob[:, idx]
                if random_s < m_prob:
                    if random_inv == 1:
                        tmp_gene_prob = gene_prob_inv[:, idx]
                    cand[idx] = tuple(self.search_space[idx][np.random.choice(len(self.search_space[idx]), p=tmp_gene_prob)])
                    
            return tuple(cand)

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            # cand = next(cand_iter)
            cand = random_func()
            if not self.is_legal(cand, 'muta' + it_idx):
                continue
            res.append(cand)
            logger.info('mutation {}/{}'.format(len(res), mutation_num))

        logger.info('mutation_num = {}'.format(len(res)))
        return res
    
    def get_crossover(self, k, crossover_num, it_idx):
        assert k in self.keep_top_k
        logger.info('crossover ......')
        res = []
        # iter = 0
        max_iters = 10 * crossover_num

        k_fitness = []
        for idv in self.keep_top_k[k]:
            k_fitness.append(self.calculate_fitness(idv))

        def random_func():
            p1 = random.choices(self.keep_top_k[k], weights=k_fitness)[0] # Rowlette Wheel Selection (prob based on fitness value)
            p2 = random.choices(self.keep_top_k[k], weights=k_fitness)[0]
            max_iters_tmp = 50
            while len(p1) != len(p2) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choices(self.keep_top_k[k], weights=k_fitness)[0]
                p2 = random.choices(self.keep_top_k[k], weights=k_fitness)[0]
            # return tuple(random.choice([i, j]) for i, j in zip(p1, p2))
            return tuple(random.choices([p1[i], p2[i]], weights=[self.acg.get_adapt_prob()[p1[i]][i] * self.acg.get_adapt_filter()[p1[i]][i], self.acg.get_adapt_prob()[p2[i]][i] * self.acg.get_adapt_filter()[p2[i]][i]])[0] for i in range(len(p1)))

        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            # cand = next(cand_iter)
            cand = random_func()
            if not self.is_legal(cand, 'cross' + it_idx):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def calculate_fitness(self, candidate):
        return self.pareto_coef * int(candidate in self.keep_top_k['pareto']) + self.acc_coef * (self.vis_dict[candidate]['acc'] - self.acc_threshold) + self.params_coef * ((self.dense_params_num - self.vis_dict[candidate]['params']) / self.dense_params_num * 100.0)
    
    def search(self):
        logger.info(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        self.get_random(self.population_num)
        logger.info(f'Probabilities of choices in current search space: {self.acg.get_adapt_prob()} {self.acg.get_adapt_filter()}')

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            acc_norm_term = max(self.vis_dict[x]['acc'] for x in self.vis_dict if 'acc' in self.vis_dict[x]) - self.acc_threshold
            params_norm_term = self.parameters_limits - self.min_parameters_limits
            self.update_pareto(self.candidates)
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.pareto_coef * int(x in self.keep_top_k['pareto']) + self.acc_coef * (self.vis_dict[x]['acc'] - self.acc_threshold) / acc_norm_term + self.params_coef * ((self.parameters_limits - self.vis_dict[x]['params']) / params_norm_term))
            self.update_top_k(
                self.candidates, k=self.population_num, key=lambda x: self.pareto_coef * int(x in self.keep_top_k['pareto']) + self.acc_coef * (self.vis_dict[x]['acc'] - self.acc_threshold) / acc_norm_term + self.params_coef * ((self.parameters_limits - self.vis_dict[x]['params']) / params_norm_term))
            
            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[self.select_num])))
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[self.select_num]):
                print('No.{} {} Top-1 val acc = {}, params = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['acc'], self.vis_dict[cand]['params']))
                tmp_accuracy.append(self.vis_dict[cand]['acc'])
            self.top_accuracies.append(tmp_accuracy)
            if self.epoch == 0:
                self.save_checkpoint()

            range_of_top_k = abs(self.vis_dict[self.keep_top_k[self.select_num][0]]['params'] - self.vis_dict[self.keep_top_k[self.select_num][self.select_num - 1]]['params'])
            adaptive_mut_prob = 1.0 - range_of_top_k / (self.parameters_limits - self.min_parameters_limits)
            mutation = self.get_mutation(
                self.select_num, self.mutation_num, (self.m_prob + adaptive_mut_prob)/2.0, (self.s_prob + adaptive_mut_prob)/2.0, str(self.epoch))
            crossover = self.get_crossover(self.select_num, self.crossover_num, str(self.epoch))

            self.candidates = mutation + crossover
            self.get_random(max(0, self.population_num - self.mutation_num - self.crossover_num))
            
            logger.info(f'Probabilities of choices in current search space (epoch {self.epoch}): {self.acg.get_adapt_prob()} {self.acg.get_adapt_filter()}')
            self.epoch += 1

            self.save_checkpoint()

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

    logger.info(str(model))

    if not args.only_cpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Set the largest by default
    if config.NAS.INIT_CONFIG is None:
        model_without_ddp.set_largest_config()
    else:
        model_without_ddp.set_sample_config(config.NAS.INIT_CONFIG)
    
    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)

    if config.NAS.PROXY.ENABLE:
        data_loader_proxy = build_proxy_loader(config)
        proxy_samples_lists = []
        
        for samples, targets in data_loader_proxy:
            proxy_samples_lists.append(samples)
        
        proxy_samples = torch.cat(proxy_samples_lists, dim = 0).cuda(non_blocking = True)
        print(
            f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build proxy tensor with shape:{proxy_samples.shape}")
    else:
        data_loader_proxy = data_loader_val

    t = time.time()

    searcher = EvolutionSearcher(args, config.NAS.SEARCH_SPACE, model, model_without_ddp, data_loader_proxy, config)

    if args.watch_checkpoints:
        print(f"checkpoint is provided {args.watch_checkpoints}, load checkpoint!")
        searcher.watch_checkpoint(args.watch_checkpoints)
    else:
        searcher.search()

    logger.info('total searching time = {:.2f} hours'.format(
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
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(config.SEED)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if is_main_process():
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())

    main(args, config)
