import numpy as np
import guided_cgs_base as cgs_base
import guided_choices as choice

def conn_mat(n_in, n_out, block_sizes, drop_ratios, w_mat, equal_blks_for_input=False, recursive_call=1):
    recursive_call = len(block_sizes)
    if recursive_call == 0:
        return np.full((n_in, n_out), 1, dtype='float32')
    block_size = block_sizes.pop()
    drop_ratio = drop_ratios.pop()
    sparsity = 1 - float(drop_ratio) / 100
    n_blk_rows = n_in // block_size
    mat_abs = w_mat
    conn_mat = np.full((n_in, n_out), 0, dtype='float32')
    if n_in % block_size != 0:
        n_blk_rows += 1
    n_blk_cols = n_out // block_size
    if n_out % block_size != 0:
        n_blk_cols += 1
    if equal_blks_for_input:
        n_blk_sels = int(round(n_blk_cols * sparsity))
        # print n_blk_sels
        for i in range(n_blk_rows-1):
            # guided_choices = np.random.choice(n_blk_cols, n_blk_sels, False)
            guided_choices = choice.guided_array_rows(mat_abs[i * block_size:(i + 1) * block_size, :], n_blk_cols, n_blk_sels, block_size)
            # print guided_choices
            for j in range(n_blk_sels):
                if guided_choices[j] == n_blk_cols-1 and n_out % block_size != 0:
                    conn_mat[i*block_size:(i+1)*block_size, guided_choices[j]*block_size:n_out] = 1
                    r_h, c_h = conn_mat[i*block_size:(i+1)*block_size, guided_choices[j]*block_size:n_out].shape
                    conn_mat[i * block_size:(i + 1) * block_size, guided_choices[j] * block_size:n_out] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], mat_abs[i * block_size:(i + 1) * block_size, guided_choices[j] * block_size:n_out], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
                else:
                    conn_mat[i*block_size:(i+1)*block_size, guided_choices[j]*block_size:(guided_choices[j]+1)*block_size] = 1
                    r_h, c_h = conn_mat[i * block_size:(i + 1) * block_size, guided_choices[j] * block_size:(guided_choices[j] + 1) * block_size].shape
                    conn_mat[i * block_size:(i + 1) * block_size, guided_choices[j] * block_size:(guided_choices[j] + 1) * block_size] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], mat_abs[i * block_size:(i + 1) * block_size, guided_choices[j] * block_size:(guided_choices[j] + 1) * block_size], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
        # guided_choices = np.random.choice(n_blk_cols, n_blk_sels, False)
        guided_choices = choice.guided_array_rows(mat_abs[(n_blk_rows - 1) * block_size:n_in, :], n_blk_cols, n_blk_sels, block_size)
        for j in range(n_blk_sels):
            conn_mat[(n_blk_rows-1)*block_size:n_in, guided_choices[j]*block_size:(guided_choices[j]+1)*block_size] = 1
            r_h, c_h = conn_mat[(n_blk_rows - 1) * block_size:n_in, guided_choices[j] * block_size:(guided_choices[j] + 1) * block_size].shape
            conn_mat[(n_blk_rows - 1) * block_size:n_in, guided_choices[j] * block_size:(guided_choices[j] + 1) * block_size] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], mat_abs[(n_blk_rows - 1) * block_size:n_in, guided_choices[j] * block_size:(guided_choices[j] + 1) * block_size], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
    else:
        n_blk_sels = int(round(n_blk_rows * sparsity))
        for i in range(n_blk_cols-1):
            guided_choices = np.random.choice(n_blk_rows, n_blk_sels, False)
            for j in range(n_blk_sels):
                if guided_choices[j] == n_blk_rows-1 and n_in % block_size != 0:
                    conn_mat[guided_choices[j]*block_size:n_in, i*block_size:(i+1)*block_size] = 1
                    r_h, c_h = conn_mat[guided_choices[j] * block_size:n_in, i * block_size:(i + 1) * block_size].shape
                    conn_mat[guided_choices[j] * block_size:n_in, i * block_size:(i + 1) * block_size] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
                else:
                    conn_mat[guided_choices[j]*block_size:(guided_choices[j]+1)*block_size, i*block_size:(i+1)*block_size] = 1
                    r_h, c_h = conn_mat[guided_choices[j] * block_size:(guided_choices[j] + 1) * block_size, i * block_size:(i + 1) * block_size].shape
                    conn_mat[guided_choices[j] * block_size:(guided_choices[j] + 1) * block_size, i * block_size:(i + 1) * block_size] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
        guided_choices = np.random.choice(n_blk_rows, n_blk_sels, False)
        for j in range(n_blk_sels):
            conn_mat[guided_choices[j]*block_size:(guided_choices[j]+1)*block_size, (n_blk_cols-1)*block_size:n_out] = 1
            r_h, c_h = conn_mat[guided_choices[j] * block_size:(guided_choices[j] + 1) * block_size, (n_blk_cols - 1) * block_size:n_out].shape
            conn_mat[guided_choices[j] * block_size:(guided_choices[j] + 1) * block_size, (n_blk_cols - 1) * block_size:n_out] = cgs_base.conn_mat(r_h, c_h, block_sizes[:], drop_ratios[:], equal_blks_for_input=equal_blks_for_input, recursive_call=recursive_call)
    return conn_mat