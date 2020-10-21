import numpy as np

def conn_mat(n_in, n_out, block_size, drop_ratio, equal_blks_for_input = False):
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
        # print n_blk_sels
        for i in range(n_blk_rows-1):
            rand_choices = np.random.choice(n_blk_cols, n_blk_sels, False)
            # print rand_choices
            for j in range(n_blk_sels):
                if rand_choices[j] == n_blk_cols-1 and n_out % block_size != 0:
                    conn_mat[i*block_size:(i+1)*block_size, rand_choices[j]*block_size:n_out] = 1
                else:
                    conn_mat[i*block_size:(i+1)*block_size, rand_choices[j]*block_size:(rand_choices[j]+1)*block_size] = 1
        rand_choices = np.random.choice(n_blk_cols, n_blk_sels, False)
        for j in range(n_blk_sels):
            conn_mat[(n_blk_rows-1)*block_size:n_in, rand_choices[j]*block_size:(rand_choices[j]+1)*block_size] = 1
    else:
        n_blk_sels = int(round(n_blk_rows * sparsity))
        for i in range(n_blk_cols-1):
            rand_choices = np.random.choice(n_blk_rows, n_blk_sels, False)
            for j in range(n_blk_sels):
                if rand_choices[j] == n_blk_rows-1 and n_in % block_size != 0:
                    conn_mat[rand_choices[j]*block_size:n_in, i*block_size:(i+1)*block_size] = 1
                else:
                    conn_mat[rand_choices[j]*block_size:(rand_choices[j]+1)*block_size, i*block_size:(i+1)*block_size] = 1
        rand_choices = np.random.choice(n_blk_rows, n_blk_sels, False)
        for j in range(n_blk_sels):
            conn_mat[rand_choices[j]*block_size:(rand_choices[j]+1)*block_size, (n_blk_cols-1)*block_size:n_out] = 1
    return conn_mat