import torch
import numpy as np

def guided_array_rows (w_mat, n_blk, n_blk_sels, blk_size):

    # garray = np.random.choice(n_blk, n_blk_sels, False)
    r, c = w_mat.shape
    temp = torch.randn(1, 1, r, c)
    temp[0, 0, :, :] = w_mat
    if r == blk_size:
        avg = torch.nn.AvgPool2d(blk_size, blk_size)
    else:
        avg = torch.nn.AvgPool2d((r, blk_size), blk_size)
    out = avg(temp)
    if c % blk_size != 0:
        x = ((n_blk-1) * blk_size)
        c1 = c - x
        w_mat2 = w_mat[: ,x:x + c1]
        temp = torch.randn(1, 1, r, c1)
        temp[0, 0, :, :] = w_mat2
        if r == blk_size:
            avg = torch.nn.AvgPool2d(blk_size, c1)
        else:
            avg = torch.nn.AvgPool2d((r, c1), c1)
        out2 = avg(temp)
        out = torch.cat((out,out2),3)

    K = out[0,0,:,:].numpy()
    garray = np.argsort(K[0])[-n_blk_sels:]

    return garray