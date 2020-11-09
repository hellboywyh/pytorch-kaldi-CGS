'''
Descripttion: 
version: 
Author: Wang Yanhong
email: 284520535@qq.com
Date: 2020-10-20 06:22:02
LastEditors: Please set LastEditors
LastEditTime: 2020-11-08 12:41:29
'''

import numpy as np
import os
from sparsity import sparsity
from sparsity import write_excel


def pattern_certain_nnz_prun_model(models, pattern_num, pattern_shape, pattern_nnz, if_pattern_prun):
    lstm_pattern = True
    mlp_pattern = True
    print(models)
    for layers_name in models:
        models[layers_name], _ = sparsity.find_pattern_certain_nnz_model(
            models[layers_name], pattern_num, pattern_shape, pattern_nnz, if_pattern_prun)
    return models


def pattern_coding(nns, outfolder, pattern_num, pattern_shape, prun_pattern_num=16, coo_threshold=4, if_pattern_prun=False):
    excel_name = "./sparsity/lstm.xls"
    # sparsity_rate = prune_perc
    for net in nns.keys():
        sparsity_rate = nns[net].prune_perc[0]*0.01
        patterns = dict()
        name_list = list()
        para_list = list()
        model = nns[net]
        for name, para in model.named_parameters():
            if not para.dim() == 1:
                name_list.append(name)
                para_list.append(para)
                print(name, para.size())

        a = model.state_dict()
        zero_cnt = 0
        all_cnt = 0
        for i, name in enumerate(name_list):
            raw_w = para_list[i]
            if raw_w.size(1)==440:
                pattern_shape = [64,8]
            elif raw_w.size(0)==48 or raw_w.size(0)==1928:
                pattern_shape = [8,64]
            else:
                pattern_shape = [16,32]
            patterns, pattern_match_num, pattern_coo_nnz, pattern_nnz, pattern_inner_nnz = \
                sparsity.find_pattern_by_similarity(
                    raw_w, pattern_num, pattern_shape, sparsity_rate, coo_threshold)
            pattern_num_memory_dict, pattern_num_cal_num_dict, pattern_num_coo_nnz_dict = \
                sparsity.pattern_curve_analyse(raw_w.size(
                ), pattern_shape, patterns, pattern_match_num, pattern_coo_nnz, pattern_nnz, pattern_inner_nnz)
            write_excel.write_pattern_curve_analyse_lstm(f"./sparsity/sparsity_{int(1/(1-sparsity_rate))}x/{net}.xls", name, f'{net}_{name}_{list(raw_w.size())}_{pattern_shape}_{coo_threshold}',
                                                         patterns, pattern_match_num, pattern_coo_nnz, pattern_nnz, pattern_inner_nnz, pattern_num_memory_dict, pattern_num_cal_num_dict, pattern_num_coo_nnz_dict)

            # if if_pattern_prun:
            # a[name] = raw_w * mask.unsqueeze(2)
            # pattern_prun_layer(raw_w, patterns, pattern_bitmaps, prun_pattern_num)
        # model.load_state_dict(a)


def pattern_prun_layer(raw_w, patterns, pattern_bitmaps, pattern_num=16):
    return raw_w


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
