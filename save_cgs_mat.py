import numpy as np
import scipy.io as sio
import cgs_base
import torch
import os

def save_mat(in_mat, mat_num = '1', dir='/home/dkadetot/pytorch-kaldi/saved_mat'):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    in_mat = in_mat.cpu()
    in_mat_num = in_mat.numpy()
    sio.savemat(dir + '/mat%s.mat' % mat_num, {'M%s' % mat_num: in_mat_num})
    return 1

def save_hcgs_mat(in_mat, mat_num = '1', dir='/home/dkadetot/pytorch-kaldi/saved_mat'):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    in_mat = in_mat.cpu()
    in_mat_num = in_mat.numpy()
    sio.savemat(dir + '/conn_mat%s.mat' % mat_num, {'CM%s' % mat_num: in_mat_num})
    return 1