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


class Pattern(Module):
    """Creates HCGS layer

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    Attributes:
        mask: the non-learnable weights of the module of shape
            `(out_features x in_features)`
    """

    def __init__(self, dense_features, pattern_shape, pattern_nnz, pattern_num, save_dir, pattern_from_file=False):
        super(Pattern, self).__init__()
        self.dense_features = dense_features
        self.pattern_shape = np.array(pattern_shape)
        self.pattern_nnz = int(pattern_nnz)
        self.pattern_num = int(pattern_num)
        self.pattern = self.get_pattern(pattern_from_file)
        self.mask = self.pattern_mask()


    def get_pattern(self, pattern_from_file=False):
        # block_num = self.dense_features//self.block_size
        if pattern_from_file:
            if not os.path.exists(pattern_from_file):
                print("Error: Pattern file \"%s\" does not exists!"%pattern_from_file)
                exit()
            pattern = np.load(pattern_from_file)
            pattern = np.split(pattern,self.pattern_num,0)
        else:
            pattern = []
            for i in range(self.pattern_num):
                mask = np.zeros(np.prod(self.pattern_shape))
                mask[np.random.choice(mask.shape[0],self.pattern_nnz)] = 1
                pattern.append(mask.reshape(self.pattern_shape))
        return pattern

    def pattern_mask(self):
        assert self.dense_features.shape[0]%self.pattern_shape[0]==0, f'Error:{self.dense_features.shape[0]} can not be divisible by {self.pattern_shape[0]}'
        self.mask = np.zeros(self.dense_features.shape)
        row_block_num = self.dense_features.shape[0]//self.pattern_shape[0]
        col_block_num = self.dense_features.shape[1]//self.pattern_shape[1]
        for i in range(row_block_num):
            for j in range(col_block_num):
                self.mask[i*self.pattern_shape[0]:(i+1)*self.pattern_shape[0],\
                    j*self.pattern_shape[1]:(j+1)*self.pattern_shape[1]] = \
                    self.pattern[np.random.choice(self.pattern_num,1)[0]]
        return(Parameter(torch.from_numpy(self.mask)))

    def update(self, dense_features):
        self.dense_features = dense_features
        self.mask = Parameter(hcgs.conn_mat(out_features, in_features, block_sizes[:], drop_ratios[:], des))

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