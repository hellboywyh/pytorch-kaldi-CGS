import torch
import numpy as np

def guided_array_rows (w_mat, n_blk, n_blk_sels, blk_size):

    # garray = np.random.choice(n_blk, n_blk_sels, False)
    r_h, c_h = w_mat.shape
    temp = torch.randn(1,1,r_h,c_h)
    temp[0,0,:,:] = w_mat
    if r_h == blk_size:
        avg = torch.nn.AvgPool2d(blk_size,blk_size)
    else:
        avg = torch.nn.AvgPool2d((r_h,blk_size),blk_size)
    out = avg(temp)
    K = out[0,0,:,:].numpy()
    garray = np.argsort(K[0])[-n_blk_sels:]

    return garray