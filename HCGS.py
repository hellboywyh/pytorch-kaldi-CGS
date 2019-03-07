import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.module import Module
import hcgs


class HCGS(Module):
    r"""Creates HCGS layer

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    Attributes:
        mask: the non-learnable weights of the module of shape
            `(out_features x in_features)`
    """

    def __init__(self, in_features, out_features, block_size1, drop_ratio1, block_size2, drop_ratio2, des='xyz'):
        super(HCGS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mask = Parameter(hcgs.conn_mat(out_features, in_features, block_size1, drop_ratio1, block_size2, drop_ratio2, des))  # torch.Tensor(, ))
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.mask.size(1))
    #     self.mask.data.uniform_(-stdv, stdv)
    #
    # def forward(self, input):
    #     return F.linear(input, self.weight, self.bias)
    #
    # def extra_repr(self):
    #     return 'in_features={}, out_features={}, bias={}'.format(
    #         self.in_features, self.out_features, self.bias is not None
    #     )
