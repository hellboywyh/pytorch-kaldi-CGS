import numpy as np
import scipy.io as sio
import cgs_base
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn

def conn_mat(n_in, n_out, block_sizes, drop_ratios, mat_num='1', dir='/home/dkadetot/saved_mat', equal_blks_for_input = True, for_test = False):
    if not len(block_sizes) == len(drop_ratios):
        print('block size and drop ratio should have the same length!')
        exit()
    block_sizes.reverse()
    drop_ratios.reverse()
    recursive_call = len(block_sizes)
    block_size = block_sizes.pop()
    drop_ratio = drop_ratios.pop()
    sparsity = 1 - float(drop_ratio) / 100
    n_blk_rows = n_in / block_size
    conn_mat = np.full((n_in, n_out), 0, dtype='float32')
    if n_in % block_size != 0:
        n_blk_rows += 1
    n_blk_cols = n_out / block_size
    if n_out % block_size != 0:
        n_blk_cols += 1
    if equal_blks_for_input:
        n_blk_sels = int(round(n_blk_cols * sparsity))
        n_blk_rows = int(n_blk_rows)
        n_blk_cols = int(n_blk_cols)
        for i in range(n_blk_rows-1):
            rand_choices = np.random.choice(n_blk_cols, n_blk_sels, False)
            # print rand_choices
            for j in range(n_blk_sels):
                if rand_choices[j] == n_blk_cols-1 and n_out % block_size != 0:
                    conn_mat[i*block_size:(i+1)*block_size, rand_choices[j]*block_size:n_out] = 1
                    r_h, c_h = conn_mat[i*block_size:(i+1)*block_size, rand_choices[j]*block_size:n_out].shape
                    conn_mat[i * block_size:(i + 1) * block_size, rand_choices[j] * block_size:n_out] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
                else:
                    conn_mat[i*block_size:(i+1)*block_size, rand_choices[j]*block_size:(rand_choices[j]+1)*block_size] = 1
                    r_h, c_h = conn_mat[i * block_size:(i + 1) * block_size, rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size].shape
                    conn_mat[i * block_size:(i + 1) * block_size, rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
        rand_choices = np.random.choice(n_blk_cols, n_blk_sels, False)
        for j in range(n_blk_sels):
            conn_mat[(n_blk_rows-1)*block_size:n_in, rand_choices[j]*block_size:(rand_choices[j]+1)*block_size] = 1
            r_h, c_h = conn_mat[(n_blk_rows - 1) * block_size:n_in, rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size].shape
            conn_mat[(n_blk_rows - 1) * block_size:n_in, rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
    else:
        n_blk_sels = int(round(n_blk_rows * sparsity))
        for i in range(n_blk_cols-1):
            rand_choices = np.random.choice(n_blk_rows, n_blk_sels, False)
            for j in range(n_blk_sels):
                if rand_choices[j] == n_blk_rows-1 and n_in % block_size != 0:
                    conn_mat[rand_choices[j]*block_size:n_in, i*block_size:(i+1)*block_size] = 1
                    r_h, c_h = conn_mat[rand_choices[j] * block_size:n_in, i * block_size:(i + 1) * block_size].shape
                    conn_mat[rand_choices[j] * block_size:n_in, i * block_size:(i + 1) * block_size] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
                else:
                    conn_mat[rand_choices[j]*block_size:(rand_choices[j]+1)*block_size, i*block_size:(i+1)*block_size] = 1
                    r_h, c_h = conn_mat[rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size, i * block_size:(i + 1) * block_size].shape
                    conn_mat[rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size, i * block_size:(i + 1) * block_size] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
        rand_choices = np.random.choice(n_blk_rows, n_blk_sels, False)
        for j in range(n_blk_sels):
            conn_mat[rand_choices[j]*block_size:(rand_choices[j]+1)*block_size, (n_blk_cols-1)*block_size:n_out] = 1
            r_h, c_h = conn_mat[rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size, (n_blk_cols - 1) * block_size:n_out].shape
            conn_mat[rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size, (n_blk_cols - 1) * block_size:n_out] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
    if for_test:
        # Save conn_mat in mat file
        sio.savemat(dir + '/conn_mat%s.mat' % mat_num, {'CM%s' % mat_num: conn_mat})
        return conn_mat
    else:
        conn_mat_torch = torch.Tensor(n_in, n_out)
        conn_mat_torch = torch.from_numpy(conn_mat)
        device = torch.device("cuda")
        conn_mat_torch = conn_mat_torch.to(device)
        conn_mat_torch.requires_grad_(False)
        return conn_mat_torch


def conn_mat(n_in, n_out, block_sizes, drop_ratios, mat_num='1', dir='/home/dkadetot/saved_mat', equal_blks_for_input = True, for_test = False):
    if not len(block_sizes) == len(drop_ratios):
        print('block size and drop ratio should have the same length!')
        exit()
    block_sizes.reverse()
    drop_ratios.reverse()
    recursive_call = len(block_sizes)
    block_size = block_sizes.pop()
    drop_ratio = drop_ratios.pop()
    sparsity = 1 - float(drop_ratio) / 100
    n_blk_rows = n_in / block_size
    conn_mat = np.full((n_in, n_out), 0, dtype='float32')
    if n_in % block_size != 0:
        n_blk_rows += 1
    n_blk_cols = n_out / block_size
    if n_out % block_size != 0:
        n_blk_cols += 1
    if equal_blks_for_input:
        n_blk_sels = int(round(n_blk_cols * sparsity))
        n_blk_rows = int(n_blk_rows)
        n_blk_cols = int(n_blk_cols)
        for i in range(n_blk_rows-1):
            rand_choices = np.random.choice(n_blk_cols, n_blk_sels, False)
            # print rand_choices
            for j in range(n_blk_sels):
                if rand_choices[j] == n_blk_cols-1 and n_out % block_size != 0:
                    conn_mat[i*block_size:(i+1)*block_size, rand_choices[j]*block_size:n_out] = 1
                    r_h, c_h = conn_mat[i*block_size:(i+1)*block_size, rand_choices[j]*block_size:n_out].shape
                    conn_mat[i * block_size:(i + 1) * block_size, rand_choices[j] * block_size:n_out] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
                else:
                    conn_mat[i*block_size:(i+1)*block_size, rand_choices[j]*block_size:(rand_choices[j]+1)*block_size] = 1
                    r_h, c_h = conn_mat[i * block_size:(i + 1) * block_size, rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size].shape
                    conn_mat[i * block_size:(i + 1) * block_size, rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
        rand_choices = np.random.choice(n_blk_cols, n_blk_sels, False)
        for j in range(n_blk_sels):
            conn_mat[(n_blk_rows-1)*block_size:n_in, rand_choices[j]*block_size:(rand_choices[j]+1)*block_size] = 1
            r_h, c_h = conn_mat[(n_blk_rows - 1) * block_size:n_in, rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size].shape
            conn_mat[(n_blk_rows - 1) * block_size:n_in, rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
    else:
        n_blk_sels = int(round(n_blk_rows * sparsity))
        for i in range(n_blk_cols-1):
            rand_choices = np.random.choice(n_blk_rows, n_blk_sels, False)
            for j in range(n_blk_sels):
                if rand_choices[j] == n_blk_rows-1 and n_in % block_size != 0:
                    conn_mat[rand_choices[j]*block_size:n_in, i*block_size:(i+1)*block_size] = 1
                    r_h, c_h = conn_mat[rand_choices[j] * block_size:n_in, i * block_size:(i + 1) * block_size].shape
                    conn_mat[rand_choices[j] * block_size:n_in, i * block_size:(i + 1) * block_size] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
                else:
                    conn_mat[rand_choices[j]*block_size:(rand_choices[j]+1)*block_size, i*block_size:(i+1)*block_size] = 1
                    r_h, c_h = conn_mat[rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size, i * block_size:(i + 1) * block_size].shape
                    conn_mat[rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size, i * block_size:(i + 1) * block_size] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
        rand_choices = np.random.choice(n_blk_rows, n_blk_sels, False)
        for j in range(n_blk_sels):
            conn_mat[rand_choices[j]*block_size:(rand_choices[j]+1)*block_size, (n_blk_cols-1)*block_size:n_out] = 1
            r_h, c_h = conn_mat[rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size, (n_blk_cols - 1) * block_size:n_out].shape
            conn_mat[rand_choices[j] * block_size:(rand_choices[j] + 1) * block_size, (n_blk_cols - 1) * block_size:n_out] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
    if for_test:
        # Save conn_mat in mat file
        sio.savemat(dir + '/conn_mat%s.mat' % mat_num, {'CM%s' % mat_num: conn_mat})
        return conn_mat
    else:
        conn_mat_torch = torch.Tensor(n_in, n_out)
        conn_mat_torch = torch.from_numpy(conn_mat)
        device = torch.device("cuda")
        conn_mat_torch = conn_mat_torch.to(device)
        conn_mat_torch.requires_grad_(False)
        return conn_mat_torch