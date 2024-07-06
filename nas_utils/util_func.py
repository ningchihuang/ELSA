import math
import random
import heapq
import copy
import numpy as np

from collections import defaultdict

class AdaptiveCandGenerator: # Werewolf (guess the wolf and give it lower probability to be chosen)
    def __init__(self, sparsity_config, training_mode):
        self.sparsity_config = sparsity_config
        self.config_length = len(sparsity_config) # num of choice blocks, e.g., [[1, 4], [2, 4], [4, 4]]
        # self.adapt_prob = {}
        self.adapt_fitness = {}
        self.num_visited = {}
        self.is_fitness_updated = False
        self.sorted_choices = {}
        self.num_filtered = 0
        self.filter = {} # 1: keep; 0: filter out
        self.training_mode = training_mode
        for config_0 in self.sparsity_config[0]: # suppose that all blocks has the same len(config_0) or max_config_index = 0
            self.adapt_fitness[tuple(config_0)] = [1000.0 * float(config_0[0]) / float(config_0[1])] * self.config_length # avoid to get sparse_level that is sparser and low-accurate in early selection (they may cause all combinations have low scores)
            self.filter[tuple(config_0)] = [1] * self.config_length
            self.num_visited[tuple(config_0)] = [0] * self.config_length
        tmp_sparsity_config = []
        for block_i in self.sparsity_config:
            tmp_block = []
            for config_i in block_i:
                tmp_block.append(tuple(config_i))
            tmp_sparsity_config.append(tmp_block)
        self.sparsity_config = tmp_sparsity_config
        # print(self.sparsity_config)
        # print(self.adapt_fitness)

    def get_adapt_prob(self, layer_i = None):
        adapt_prob = {}
        tmp_A = np.array(list(self.adapt_fitness.values()))
        if layer_i == None:
            gene_prob = tmp_A / np.sum(list(tmp_A), axis=0)
            for choice_i in self.sparsity_config[0]:
                adapt_prob[choice_i] = [0] * self.config_length
                adapt_prob[choice_i] = gene_prob[list(adapt_prob.keys()).index(choice_i)]
        else:
            layer_sum = sum(tmp_A[:, layer_i])
            for choice_i in self.sparsity_config[layer_i]:
                adapt_prob[choice_i] = self.adapt_fitness[choice_i][layer_i] / layer_sum
        return adapt_prob
    
    def get_adapt_filter(self, layer_i = None):
        if layer_i == None:
            return self.filter
        else:
            adapt_filter = {}
            for choice_i in self.sparsity_config[layer_i]:
                adapt_filter[choice_i] = self.filter[choice_i][layer_i]
            return self.filter

    def update_adapt_fitness(self, cand, cand_acc):
        """
        Used after visiting (evaluating) a new cand
        Parameters:
            cand: a cand_config, a list of tuples of choices, e.g., [(1, 4), (4, 4), ..., (2, 4)]
            cand_acc: its accuracy, a float in [0.0, 1.0]
        """
        # print('update probability ......')
        for i in range(len(cand)):
            gene = cand[i]
            num_v = self.num_visited[gene][i]
            self.adapt_fitness[gene][i] = (cand_acc + self.adapt_fitness[gene][i] * num_v ) / (num_v + 1)
            self.num_visited[gene][i] += 1
        # print(self.adapt_fitness)
        self.is_fitness_updated = True

    def sort_choices(self):
        index_dict = {}
        for gene in self.adapt_fitness:
            gene_fitness = self.adapt_fitness.get(gene)
            for idx in range(self.config_length):
                index_dict[(gene, idx)] = gene_fitness[idx]
        self.sorted_choices = dict(sorted(index_dict.items(), key=lambda x: x[1]))

    def update_filter(self, num_filtered, threshold, num_vis_thres):
        if self.is_fitness_updated:
            self.sort_choices()
        if self.is_fitness_updated or self.num_filtered != num_filtered:
            # print(self.sorted_choices)
            if self.num_filtered > num_filtered:
                for gene in self.filter:
                    self.filter[gene] = [1] * self.config_length # reset
            for gene in self.filter: # filtered out choices with fitness under threshold
                for idx in range(self.config_length):
                    if self.adapt_fitness[gene][idx] < threshold and self.num_visited[gene][idx] >= num_vis_thres:
                        self.filter[gene][idx] = 0
            smallest_indices = 0
            for choice_i in self.sorted_choices:
                if smallest_indices >= num_filtered:
                    break
                if self.filter[choice_i[0]][choice_i[1]] == 0:
                    smallest_indices +=1
                    continue
                num_remained_choices = 0
                for gene in self.filter:
                    if self.filter[gene][choice_i[1]] == 1:
                        num_remained_choices += 1
                if num_remained_choices > 1:
                    self.filter[choice_i[0]][choice_i[1]] = 0 # filtered out k smallest choices
                    smallest_indices +=1
                    # print(choice_i)
            if self.training_mode:
                for gene in self.num_visited:
                    self.num_visited[gene] = [0] * self.config_length
        self.num_filtered = num_filtered
        self.is_fitness_updated = False

    def adaptiv_random(self, weighted_mode = 0.0):#, filter_ratio = 0.0, filter_threhold = 0.1, num_visit_threshold = 5):
        """
        Parameters:
            weighted_mode:
                1.0: give sparser choices larger prob
                0.0: (default) equal prob, only depent on fitness scores
                -1.0: give denser choices larger prob
            filter_ratio: 0.0~0.5
                filter out how much ratio of low-score choices in the search space
        Returns:
            a cand_config, according to adapt_fitness, weighted_mode and filter_ratio
        """
        # if filter_ratio >= 0.0 and filter_ratio <= 1.0:
        #     K = int(self.config_length * (len(self.adapt_fitness) - 1) * filter_ratio)
        #     if K > 0 or filter_threhold > 0.0:
        #         self.update_filter(K, filter_threhold, num_visit_threshold)
        # else:
        #     raise ValueError(f"Expect `filter_ratio` a float in [0.0, 1.0]")

        res = []
        for idx in range(self.config_length):
            weighted = []
            for config_i in self.sparsity_config[idx]:
                weighted.append(pow(float(config_i[1]) / config_i[0], weighted_mode))
            weighted = np.array(weighted)
            fitness = np.array([self.adapt_fitness.get(gene)[idx] for gene in self.adapt_fitness])
            filtered = np.array([self.filter.get(gene)[idx] for gene in self.filter])
            weighted_fitness = np.multiply(np.multiply(weighted, fitness), filtered)
            res.extend(random.choices(self.sparsity_config[idx], weights=list(weighted_fitness)))
        return res # return a cand_config

class RandomCandGenerator:
    def __init__(self, sparsity_config):
        self.sparsity_config               = sparsity_config
        self.num_candidates_per_block = len(sparsity_config[0]) # might have bug if each block has different number of choices
        self.config_length            = len(sparsity_config)    # e.g., the len of DeiT-S is 48 (12 blocks, each has qkv, fc1, fc2, and linear projection)
        self.m = defaultdict(list)        # m: the magic dictionary with {index: cand_config}
        #random.seed(seed)
        v = []                            # v: a temp vector for function rec()
        self.rec(v, self.m)
        
    def calc(self, v):                    # generate the unique index for each candidate
        res = 0
        for i in range(self.num_candidates_per_block):
            res += i * v[i]
        return res

    def rec(self, v, m, idx=0, cur=0):    # recursively enumerate all possible candidates and attach unique indexes for them
        if idx == (self.num_candidates_per_block-1) :
            v.append(self.config_length - cur)
            m[self.calc(v)].append(copy.copy(v))
            v.pop()
            return

        i = self.config_length - cur
        while i >= 0:
            v.append(i)
            self.rec(v, m, idx+1, cur+i)
            v.pop()
            i -= 1
            
    def random(self):                     # generate a random index and return its corresponding candidate
        row = random.choice(random.choice(self.m))
        ratios = []
        for num, ratio in zip(row, [i for i in range(self.num_candidates_per_block)]):
            ratios += [ratio] * num
        random.shuffle(ratios)
        res = []
        for idx, ratio in enumerate(ratios):
            res.append(tuple(self.sparsity_config[idx][ratio])) # Fixme: 
        return res                        # return a cand_config

def convert_to_hashable(entry):
    """
    Convert an unhashable object (e.g., 2D list) to a hashable one (e.g., tuple of tuples).

    Parameters:
        entry: The unhashable object to be converted.

    Returns:
        object: The converted hashable object.
    """
    if isinstance(entry, list):
        return tuple(map(convert_to_hashable, entry))
    return entry


def convert_from_hashable(entry):
    """
    Convert a hashable object (e.g., tuple of tuples) back to its original unhashable form (e.g., 2D list).

    Parameters:
        entry: The hashable object to be converted.

    Returns:
        object: The converted unhashable object.
    """
    if isinstance(entry, tuple):
        return list(map(convert_from_hashable, entry))
    return entry

class CandidatePool:
    def __init__(self, candidate_pool_size=1000):
        """
        Object for candidate pools

        Parameters:
            candidate_pool_size (int, optional): The maximum size of the promising pools (max-heap). Default is 1000.
        """
        self.max_pool_size = candidate_pool_size
        self.candidate_pools = {}


    def get_size(self):
        return len(self.candidate_pools)

    def get_one_subnet(self):
        """
        Selects a subnet
        Returns:
            The selected subnet (from `candidate_pools`) if `candidate_pools` is not empty.
            Otherwise, return None
        """
        if len(self.candidate_pools) == 0:
            return None
        else:
            return convert_from_hashable(random.choice(list(self.candidate_pools.keys())))  # Extract only the subnet (index 1)
    
    def _sort_and_limit(self):
        self.candidate_pools = dict(sorted(self.candidate_pools.items(), key=lambda item: item[1][0]))
        self.candidate_pools = dict(list(self.candidate_pools.items())[:self.max_pool_size])
    
    def add_one_subnet_with_score_and_flops(self, subnet, score, flops):
        """
        Adds a subnet to the `candidate_pools` if it belongs to the top candidates.

        Parameters:
            subnet: The subnet to add to the promising pools.
            score: The score of the subnet (used for max-heap comparison).
        """
        
        self.candidate_pools[convert_to_hashable(subnet)] = (score, flops)
        self._sort_and_limit()
    
    def clear_candidate_pools(self):
        """
        Clears the `candidate_pools`.
        """
        self.candidate_pools = {}

    def state_dict(self):
        return {
            "max_pool_size": self.max_pool_size,
            #"candidate_pools" : self.candidate_pools
            "candidate_pools" : copy.deepcopy(self.candidate_pools)
        }

    def load_state_dict(self, state_dict):
        if "max_pool_size" not in state_dict:
            raise ValueError(f"Expect `max_pool_size` in the state_dict")
        if "candidate_pools" not in state_dict:
            raise ValueError(f"Expect `candidate_pools` in the state_dict")
        self.max_pool_size = state_dict['max_pool_size']
        self.candidate_pools = state_dict['candidate_pools']

    def get_candidate_architectutes(self):
        return list(self.candidate_pools.keys())
    
    def get_candidate_architecture(self, i):
        return self.get_candidate_architectutes()[i]
    

class LinearEpsilonScheduler:
    def __init__(self, total_epochs, min_eps, max_eps):
        self.total_epochs = total_epochs
        self.min_eps = min_eps
        self.max_eps = max_eps

    def get_epsilon(self, current_epoch):
        progress = min(current_epoch / (self.total_epochs - 1), 1.0)
        eps = self.min_eps + progress * (self.max_eps - self.min_eps)
        return eps



class LinearEpsilonScheduler:
    def __init__(self, total_epochs, min_eps, max_eps, patient_epochs, fixed_epochs):
        self.total_epochs = total_epochs
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.patient_epochs = patient_epochs
        self.fixed_epochs = fixed_epochs

    def get_epsilon(self, current_epoch):
        if current_epoch < self.patient_epochs:
            return self.min_eps

        if current_epoch >= self.total_epochs - self.fixed_epochs:
            return self.max_eps

        progress = min((current_epoch - self.patient_epochs) / (self.total_epochs - self.patient_epochs - self.fixed_epochs - 1), 1.0)
        eps = self.min_eps + progress * (self.max_eps - self.min_eps)
        return eps



if __name__ == '__main__':
    # # Create the SearchSpace instance with candidate choices, epsilon, and promising pool size
    # search_space = CandidatePool(candidate_pool_size=3)

    # # Add some subnets to candidate_pools with their scores (you can do this during the search process)
    # search_space.add_one_subnet_with_score_and_flops(subnet=[[2, 4], [2, 4]], score=0.91, flops = 1)
    # search_space.add_one_subnet_with_score_and_flops(subnet=[[1, 4], [2, 4]], score=0.89, flops = 2)
    # search_space.add_one_subnet_with_score_and_flops(subnet=[[3, 4], [2, 4]], score=0.6, flops = 3)
    # search_space.add_one_subnet_with_score_and_flops(subnet=[[2, 4], [1, 3]], score=0.6, flops = 4)
    # search_space.add_one_subnet_with_score_and_flops(subnet=[[3, 4], [2, 4]], score=0.90, flops = 3)
    # # ... (add more subnets with scores)
    # print(search_space.candidate_pools)

    AdpGen = AdaptiveCandGenerator([[[1, 8], [2, 8], [3, 8], [4, 8], [8, 8]], [[1, 8], [2, 8], [3, 8], [4, 8], [8, 8]], [[1, 8], [2, 8], [3, 8], [4, 8], [8, 8]], [[1, 8], [2, 8], [3, 8], [4, 8], [8, 8]], [[1, 8], [2, 8], [3, 8], [4, 8], [8, 8]]])
    print(AdpGen.adaptiv_random(weighted_mode=-1.0))
    AdpGen.update_adapt_fitness(((1, 8), (2, 8), (2, 8), (1, 8), (4, 8)), 0.21)
    AdpGen.update_adapt_fitness(((1, 8), (3, 8), (8, 8), (2, 8), (2, 8)), 0.23)
    AdpGen.update_adapt_fitness(((1, 8), (4, 8), (2, 8), (3, 8), (8, 8)), 0.32)
    AdpGen.update_adapt_fitness(((1, 8), (4, 8), (3, 8), (4, 8), (8, 8)), 0.33)
    print(AdpGen.adaptiv_random(weighted_mode=0.0))
    AdpGen.update_adapt_fitness(((2, 8), (2, 8), (4, 8), (8, 8), (3, 8)), 0.66)
    AdpGen.update_adapt_fitness(((2, 8), (8, 8), (2, 8), (2, 8), (3, 8)), 0.78)
    AdpGen.update_adapt_fitness(((8, 8), (4, 8), (1, 8), (4, 8), (2, 8)), 0.24)
    AdpGen.update_adapt_fitness(((2, 8), (2, 8), (2, 8), (1, 8), (8, 8)), 0.25)
    print(AdpGen.adaptiv_random(weighted_mode=0.0))
    AdpGen.update_adapt_fitness(((3, 8), (1, 8), (2, 8), (2, 8), (4, 8)), 0.18)
    AdpGen.update_adapt_fitness(((3, 8), (4, 8), (1, 8), (4, 8), (2, 8)), 0.31)
    AdpGen.update_adapt_fitness(((8, 8), (1, 8), (4, 8), (3, 8), (2, 8)), 0.22)
    AdpGen.update_adapt_fitness(((3, 8), (3, 8), (2, 8), (2, 8), (3, 8)), 0.69)
    print(AdpGen.adaptiv_random(weighted_mode=1.0, filter_ratio=0.25))
    AdpGen.update_adapt_fitness(((4, 8), (2, 8), (3, 8), (4, 8), (2, 8)), 0.85)
    AdpGen.update_adapt_fitness(((4, 8), (8, 8), (1, 8), (2, 8), (3, 8)), 0.26)
    AdpGen.update_adapt_fitness(((4, 8), (1, 8), (3, 8), (3, 8), (8, 8)), 0.34)
    AdpGen.update_adapt_fitness(((4, 8), (3, 8), (8, 8), (2, 8), (2, 8)), 0.77)
    AdpGen.update_adapt_fitness(((2, 8), (8, 8), (2, 8), (2, 8), (1, 8)), 0.15)
    
    print(AdpGen.adaptiv_random(weighted_mode=1.0, filter_ratio=0.5))
    print(AdpGen.adaptiv_random(weighted_mode=1.0, filter_ratio=0.5))
    print(AdpGen.adaptiv_random(weighted_mode=1.0, filter_ratio=0.5))
    
    print(AdpGen.get_adapt_prob())
    print(AdpGen.get_adapt_prob(1))
    
