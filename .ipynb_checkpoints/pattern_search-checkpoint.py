'''
Descripttion: 
version: 
Author: Wang Yanhong
email: 284520535@qq.com
Date: 2020-10-20 06:22:02
LastEditors: Wang Yanhong
LastEditTime: 2020-10-23 10:37:00
'''

import numpy as np
import os
from sparsity import sparsity


# def read_model(pt_file_arch, pruned_pt_file_arch):
#     [nns, costs] = model_init(inp_out_dict, model, config, arch_dict, use_cuda, multi_gpu, to_do)
    
#     # optimizers initialization
#     optimizers = optimizer_init(nns, config, arch_dict)

#     # pre-training
#     for net in nns.keys():
#         pt_file_arch = config[arch_dict[net][0]]['arch_pretrain_file']

#         if pt_file_arch != 'none':
#             checkpoint_load = torch.load(pt_file_arch)
#             nns[net].load_state_dict(checkpoint_load['model_par'])
#             optimizers[net].load_state_dict(checkpoint_load['optimizer_par'])
#             optimizers[net].param_groups[0]['lr'] = float(
#                 config[arch_dict[net][0]]['arch_lr'])  # loading lr of the cfg file for pt


def pattern_prun_model(models, pattern_num, pattern_shape, pattern_nnz, if_pattern_prun):
    lstm_pattern=True
    mlp_pattern=True
    print(models)
    for layers_name in models:
        models[layers_name],_ = sparsity.find_pattern_certain_nnz_model(models[layers_name],pattern_num, pattern_shape, pattern_nnz, if_pattern_prun)
    return models


def pattern_search(dense_features, pattern_mode, pattern_shape, pattern_nnz, mask_save_dir=False, mask_name=False):
    """[summary]

    Args:
        dense_features ([array]): [description]
        pattern_mode ([string]): ["pattern","coo","pattern_coo"]
        pattern_shape ([list[2]]): [pattern block shape]
        pattern_nnz ([int]): [nonzeros in a pattern block]
        mask_save_dir (bool, optional): [Directory to save mask]. Defaults to False.
        mask_name (bool, optional): [Name of mask]. Defaults to False.

    Returns:
        [array]: [the mask of pattern prunning]
    """
    # create mask by pattern_mode
    if pattern_mode == 'pattern':
        mask = pattern_mask(dense_features, pattern_shape, pattern_nnz)
    elif pattern_mode == 'coo':
        mask = coo_mask(dense_features, pattern_shape, pattern_nnz)
    elif pattern_mode == 'pattern_coo':
        mask = pattern_mask(dense_features, pattern_shape, pattern_nnz)
    else:
        mask = np.zeros(dense_features.shape)
    # save mask
    np.save(os.path.join(mask_save_dir, mask_name+".npy"))
    return mask


def pattern_mask(dense_features, pattern_shape, pattern_nnz):
    mask = np.zeros(dense_features.shape)
    return mask

def coo_mask(dense_features, pattern_shape, pattern_nnz):
    mask = np.zeros(dense_features.shape)
    return mask

def pattern_coo_mask(dense_features, pattern_shape, pattern_nnz):
    mask = np.zeros(dense_features.shape)
    return mask