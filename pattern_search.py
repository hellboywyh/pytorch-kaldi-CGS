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

def pattern_prun_model(models):
    lstm_pattern=True
    mlp_pattern=True
    print(models)
    for layers_name in models:
        models[layers_name],_ = sparsity.find_pattern_certain_nnz_model(models[layers_name],16,[8,8],4, if_pattern_prun=True)
    return models
# def pattern_prun_model(model, pattern_mode, pattern_shape, pattern_nnz, mask_save_dir=False, mask_name=False):
#     print(model)
#     return model

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