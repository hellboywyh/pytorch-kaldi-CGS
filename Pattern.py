'''
Description: 
version: 
Author: Wang Yanhong
email: 284520535@qq.com
Date: 2020-10-20 06:22:15
LastEditors: Wang Yanhong
LastEditTime: 2020-11-09 12:28:11
'''

import math
import os
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.module import Module
import hcgs
import guided_hcgs
import scipy.io as sio
import numpy as np
from data_io import read_mat
from sparsity import sparsity


class Pattern(Module):
    """Creates HCGS layer

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    Attributes:
        mask: the non-learnable weights of the module of shape
            `(out_features x in_features)`
    """

    def __init__(self, dense_features, pattern_mode, pattern_shape, pattern_nnz, pattern_num, save_dir, pattern_from_file=False):
        super(Pattern, self).__init__()
        self.dense_features = dense_features
        self.pattern_from_file = pattern_from_file
        self.pattern_mode = pattern_mode
        self.pattern_shape = np.array(pattern_shape)
        self.pattern_nnz = int(pattern_nnz)
        self.pattern_num = int(pattern_num)
        self.mask = self.get_mask()

    def get_mask(self, dense=True):
        if self.pattern_mode == 'pattern_from_weight':
            self.pattern = self.update_pattern_by_weight()
            return(Parameter(torch.ones(self.dense_features.shape)))
        elif self.pattern_mode == 'pattern':
            return self.pattern_mask()
        elif self.pattern_mode == 'coo':
            return self.coo_mask()
        elif self.pattern_mode == 'pattern_coo':
            return self.pattern_coo_mask()
        else:
            return(Parameter(torch.ones(self.dense_features.shape)))

    def get_pattern(self, pattern_nnz):
        # block_num = self.dense_features//self.block_size
        if self.pattern_from_file:
            if not os.path.exists(pattern_from_file):
                print("Error: Pattern file \"%s\" does not exists!" %
                      pattern_from_file)
                exit()
            pattern = np.load(pattern_from_file)
            pattern = np.split(pattern, self.pattern_num, 0)
        else:
            pattern = []
            for i in range(self.pattern_num):
                mask = np.zeros(np.prod(self.pattern_shape))
                mask[np.random.choice(mask.shape[0], pattern_nnz)] = 1
                pattern.append(mask.reshape(self.pattern_shape))
        return pattern

    def pattern_mask(self):
        self.pattern = self.get_pattern(self.pattern_nnz)
        assert self.dense_features.shape[0] % self.pattern_shape[
            0] == 0, f'Error:{self.dense_features.shape[0]} can not be divisible by {self.pattern_shape[0]}'
        assert self.dense_features.shape[1] % self.pattern_shape[
            1] == 0, f'Error:{self.dense_features.shape[1]} can not be divisible by {self.pattern_shape[1]}'
        self.mask = np.zeros(self.dense_features.shape)
        row_block_num = self.dense_features.shape[0]//self.pattern_shape[0]
        col_block_num = self.dense_features.shape[1]//self.pattern_shape[1]
        for i in range(row_block_num):
            for j in range(col_block_num):
                self.mask[i*self.pattern_shape[0]:(i+1)*self.pattern_shape[0],
                          j*self.pattern_shape[1]:(j+1)*self.pattern_shape[1]] = \
                    self.pattern[np.random.choice(self.pattern_num, 1)[0]]
        return(Parameter(torch.from_numpy(self.mask)))

    def coo_mask(self):
        assert self.dense_features.shape[0] % self.pattern_shape[
            0] == 0, f'Error:{self.dense_features.shape[0]} can not be divisible by {self.pattern_shape[0]}'
        assert self.dense_features.shape[1] % self.pattern_shape[
            1] == 0, f'Error:{self.dense_features.shape[1]} can not be divisible by {self.pattern_shape[1]}'
        self.mask = np.zeros(self.dense_features.shape)
        row_block_num = self.dense_features.shape[0]//self.pattern_shape[0]
        col_block_num = self.dense_features.shape[1]//self.pattern_shape[1]
        for i in range(row_block_num):
            for j in range(col_block_num):
                dense_features_block = self.dense_features[i*self.pattern_shape[0]:(i+1)*self.pattern_shape[0],
                                                           j*self.pattern_shape[1]:(j+1)*self.pattern_shape[1]].flatten()
                mask_block = np.zeros(dense_features_block.shape)
                mask_block[np.argsort(
                    np.abs(dense_features_block))[-self.pattern_nnz:]] = 1
                self.mask[i*self.pattern_shape[0]:(i+1)*self.pattern_shape[0],
                          j*self.pattern_shape[1]:(j+1)*self.pattern_shape[1]] = mask_block.reshape(self.pattern_shape)
        return(Parameter(torch.from_numpy(self.mask)))

    def pattern_coo_mask(self):
        assert self.dense_features.shape[0] % self.pattern_shape[
            0] == 0, f'Error:{self.dense_features.shape[0]} can not be divisible by {self.pattern_shape[0]}'
        assert self.dense_features.shape[1] % self.pattern_shape[
            1] == 0, f'Error:{self.dense_features.shape[1]} can not be divisible by {self.pattern_shape[1]}'
        row_block_num = self.dense_features.shape[0]//self.pattern_shape[0]
        col_block_num = self.dense_features.shape[1]//self.pattern_shape[1]
        self.mask = np.zeros(self.dense_features.shape)
        self.pat_nnz = math.ceil(self.pattern_nnz/2)
        self.coo_nnz = self.pattern_nnz - self.pat_nnz
        self.pattern = self.get_pattern(self.pat_nnz)
        for i in range(row_block_num):
            for j in range(col_block_num):
                mask_block = self.pattern[np.random.choice(
                    self.pattern_num, 1)[0]].flatten()
                dense_features_block = self.dense_features[i*self.pattern_shape[0]:(i+1)*self.pattern_shape[0],
                                                           j*self.pattern_shape[1]:(j+1)*self.pattern_shape[1]].flatten()
                mask_block[np.argsort(np.abs(np.array(
                    dense_features_block)*(np.ones_like(mask_block) - mask_block)))[-self.coo_nnz:]] = 1
                self.mask[i*self.pattern_shape[0]:(i+1)*self.pattern_shape[0],
                          j*self.pattern_shape[1]:(j+1)*self.pattern_shape[1]] = mask_block.reshape(self.pattern_shape)
        return(Parameter(torch.from_numpy(self.mask)))

    def update(self, dense_features):
        self.dense_features = dense_features
        self.mask = Parameter(hcgs.conn_mat(
            out_features, in_features, block_sizes[:], drop_ratios[:], des))

    def read_mat(self, file):
        """ [mat] = read_mat(file_or_fd)
        Reads single kaldi matrix, supports ascii and binary.
        file_or_fd : file, gzipped file, pipe or opened file descriptor.
        """
        fd = open(file)
        try:
            binary = fd.read(2).decode()
            if binary == '\0B':
                mat = _read_mat_binary(fd)
            else:
                assert (binary == ' [')
                mat = _read_mat_ascii(fd)
        finally:
            fd.close()
        return mat

    def update_pattern_by_weight(self):
        pattern_candidates = sparsity.generate_complete_pattern_set(
            self.pattern_shape, self.pattern_nnz)
        self.pattern = sparsity.find_top_k_by_similarity(
            self.dense_features, pattern_candidates, stride=self.pattern_shape, pattern_num=self.pattern_nnz)
        for i, p in enumerate(self.pattern):
            print("top", i, len(self.pattern))
            print(p)

    def update_mask(self):
        self.mask = sparsity.apply_patterns(self.dense_features, self.pattern)

        # def update_pattern_by_weight(self, input, prune_perc):
        # assert self.dense_features.shape[0] % self.pattern_shape[
        #     0] == 0, f'Error:{self.dense_features.shape[0]} can not be divisible by {self.pattern_shape[0]}'
        # assert self.dense_features.shape[1] % self.pattern_shape[
        #     1] == 0, f'Error:{self.dense_features.shape[1]} can not be divisible by {self.pattern_shape[1]}'
        # row_block_num = self.dense_features.shape[0]//self.pattern_shape[0]
        # col_block_num = self.dense_features.shape[1]//self.pattern_shape[1]
        # pattern_candidates = list()
        # pattern_scores = list()
        # pattern_len = self.pattern_shape[0]*self.pattern_shape[1]
        # for i in range(pattern_len):
        #     for j in range(pattern_len):
        #         if not i == j:
        #             pattern = torch.zeros(self.pattern_shape).flatten()
        #             pattern[i] = 1
        #             pattern[j] = 1
        #             pattern_score = cal_pattern_score(input, pattern)
        #             pattern_candidates.append(pattern)
        #             pattern_scores.append(pattern_score)
        # sorted_patterns = sorted(
        #     pattern_candidates, key=lambda i: pattern_scores[i], reverse=True)

    # def cal_pattern_score(self, input, pattern):
    #     pattern_score = 0
    #     assert self.dense_features.shape[0] % self.pattern_shape[
    #         0] == 0, f'Error:{self.dense_features.shape[0]} can not be divisible by {self.pattern_shape[0]}'
    #     assert self.dense_features.shape[1] % self.pattern_shape[
    #         1] == 0, f'Error:{self.dense_features.shape[1]} can not be divisible by {self.pattern_shape[1]}'
    #     row_block_num = self.dense_features.shape[0]//self.pattern_shape[0]
    #     col_block_num = self.dense_features.shape[1]//self.pattern_shape[1]
    #     for i in range(row_block_num):
    #         for j in range(col_block_num):
    #             pattern_score += pattern * input[i*self.pattern_shape[0]:(i+1)*self.pattern_shape[0],
    #                                              j*self.pattern_shape[1]:(j+1)*self.pattern_shape[1]].flatten()
    #     return pattern_score


# for i in range(16):
#     mask = np.zeros(64)
#     mask[i*4:(i+1)*4]=1
#     pattern.append(mask.reshape(8,8))
# np.save('pattern_file/b08b08_k04_n16_pattern_v1.npy',np.concatenate(pattern,0))
# pattern=[]
# for i in range(16):
#     mask = np.zeros(64)
#     # mask[i*4:(i+1)*4]=1
#     start = int(np.random.choice(60,1))
#     mask[start:start+4]=1
#     pattern.append(mask.reshape([8,8]))
# np.save('pattern_file/b08b08_k04_n16_pattern_v2.npy',np.concatenate(pattern,0))
