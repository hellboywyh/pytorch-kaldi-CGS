import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F
import save_cgs_mat

import numpy as np


def prune (model, pruning_perc):
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = p.data.abs() > threshold
            masks.append(pruned_inds.float())
    return masks

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def find_mean(tensor):
    sparse_tens = tensor.to_sparse()
    return sparse_tens.values().abs().mean()


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def Quantize(tensor, numBits=3, if_forward=False, balanced=True):
    # tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    tensor.clamp_(-1,1)
    tensor_sign = tensor.sign()
    if balanced:
        mean = find_mean(tensor)
        scale = mean * 2.5
        if if_forward:
            tensor.abs_().mul_(2 ** (numBits - 1)).ceil_().div_(2 ** (numBits - 1))
        else:
            tensor = tensor.abs().div(scale).mul(2 ** (numBits - 1)).ceil().mul(scale).ceil().div(2 ** (numBits - 1))
        tensor.clamp_(-1, 1)
        tensor.mul_(tensor_sign)
    else:
        # tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
        if if_forward:
            tensor.abs_().mul_(2**(numBits-1)).ceil_().div_(2**(numBits-1))
        else:
            tensor=tensor.abs().mul(2**(numBits-1)).ceil().div(2**(numBits-1))
        tensor.mul_(tensor_sign)
    return tensor

def Quantize_inp(inp, Bits, if_forward=False):
    max = inp.max()
    min = inp.min()
    max.abs_()
    min.abs_()
    if max > min:
        var = max
    else:
        var = min
    if not (var == 0.):
        inp_sign = inp.sign()
        if if_forward:
            inp.div_(var)
            inp.abs_().mul_(2 ** (Bits - 1)).ceil_().div_(2 ** (Bits - 1))
            inp.mul_(var)
        else:
            inp = inp.div(var)
            inp = inp.abs().mul(2 ** (Bits - 1)).ceil().div(2 ** (Bits - 1))
            inp = inp.mul(var)
        inp.mul_(inp_sign)
    return inp

def QuantizeVar(tensor,quant_mode='det',  params=None, numBits=3):
    max = tensor.max()
    min = tensor.min()
    tensor.clamp_(min,max)
    # min.abs_()
    # if max > min:
    #     var = max
    # else:
    #     var = min
    tensor_sign = tensor.sign()
    # tensor.div_(var)
    if quant_mode=='det':
        # tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
        tensor = tensor.abs().mul(2**(numBits-1)).ceil().div(2**(numBits-1))
        tensor.mul_(tensor_sign)
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

#import torch.nn._functions as tnnf

class BinarizeLinear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(BinarizeLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight_org = torch.Tensor(out_features, in_features)
        # self.weight_quant = self.weight.detach().clone()
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
    # def forward(self, input, i, xorh='x'):
        self.weight_org = self.weight.data
        # save_cgs_mat.save_mat(self.weight.data, str(i) + xorh + 'testw')
        self.weight.data = Binarize(self.weight.data)
        # save_cgs_mat.save_mat(self.weight.data, str(i) + xorh + 'testwb')
        out = F.linear(input, self.weight, self.bias)
        self.weight.data = self.weight_org
        # save_cgs_mat.save_mat(self.weight.data, str(i) + xorh + 'orgw')
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


# class QuantizeLinear(Module):
class QuantizeLinear(nn.Module):
    
    def __init__(self, in_features, out_features, numBits=8, bias=True, if_forward=False, if_inp_quant=False, inp_quant=16):
        super(QuantizeLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight_org = torch.Tensor(out_features, in_features)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.numBits = numBits
        self.if_forward = if_forward
        self.if_inp_quant = if_inp_quant
        self.inp_quant = inp_quant
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.if_forward:
            self.weight.data = Quantize(self.weight.data, numBits=self.numBits, if_forward=self.if_forward, balanced=False)
            # self.weight.data = QuantizeVar(self.weight.data, numBits=self.numBits)
            if self.if_inp_quant:
                input.data = Quantize_inp(input.data, self.inp_quant, self.if_forward)
            out = F.linear(input, self.weight, self.bias)
        else:
            self.weight_org = self.weight.data
            self.weight.data = Quantize(self.weight.data, numBits=self.numBits, balanced=False)
            # self.weight.data = QuantizeVar(self.weight.data, numBits=self.numBits)
            if self.if_inp_quant:
                input.data = Quantize_inp(input.data, self.inp_quant)
            out = F.linear(input, self.weight, self.bias)
            self.weight.data = self.weight_org
        return out

    # def forward(self, input):
    #     if self.if_forward:
    #         self.weight.data = Quantize(self.weight.data, numBits=self.numBits, if_forward=self.if_forward, balanced=False)
    #         # self.weight.data = QuantizeVar(self.weight.data, numBits=self.numBits)
    #         # if self.if_inp_quant:
    #         #     input.data = Quantize_inp(input.data, self.inp_quant, self.if_forward)
    #         out = F.linear(input, self.weight, self.bias)
    #     else:
    #         self.weight_org = self.weight.data
    #         # self.weight.data = Quantize(self.weight.data, numBits=self.numBits, balanced=False)
    #         # # self.weight.data = QuantizeVar(self.weight.data, numBits=self.numBits)
    #         # if self.if_inp_quant:
    #         #     input.data = Quantize_inp(input.data, self.inp_quant)
    #         # out = F.linear(input, self.weight, self.bias)
    #         self.weight.data = self.weight_org
    #     return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
