[TOC]
### 1. 
```
 File "/yhwang/0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/HCGS.py", line 58, in __init__
    self.mask = Parameter(guided_hcgs.conn_mat(out_features, in_features, block_sizes[:], drop_ratios[:], w_mat, des))  # torch.Tensor(, ))
  File "/yhwang/0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/guided_hcgs.py", line 32, in conn_mat
    guided_choices = choice.guided_array_rows(mat_abs[i*block_size:(i+1)*block_size,:], n_blk_cols, n_blk_sels, block_size)
  File "/yhwang/0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/guided_choices.py", line 18, in guided_array_rows
    w_mat2 = w_mat[: ,x:x + c1]
TypeError: slice indices must be integers or None or have an __index__ method
```
都是python2 python3问题导致，把"/"换成"//"即可

### 2.
量化：
Binary:
tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)
这里teensor是linear.weight.data
inps:
Quantized_models.py(line96) quantised_inp 

### 3.20200909开会
* CGS实现：
    * (1)baseline 32-bit 512-cell LSTM:16.6%PER
    * (2)+ quantization 6-bit weight 13-bit activation(16x compression): 18.7%PER 
    * (3)+ hcgs(16x compression): 20.6%(图中看出)（hardware design point）
    * (4)+ 其实还做了一步让四个gate的structure相同:据说只是会增加0.2%的PER
* 实验:
    * (1)16.4%
    * (2):16.5% 只有LSTM进行了quant；
    * (3):18.1% 只有LSTM进行了quant,hcgs；
    * (4):17.7% LSTM+2MLP都进行了quant,hcgs;
    * (5):16.9% LSTM+2MLP都进行了quant;
    见：model.svg
    //20200910 update
    * (5):18.2% 8bweight_16bactivate+hcgs
    * (6):16.6% 8bweight_16bactivate

    
* 压缩代码实现：
    * (1)quantization:
    大概就是先scale到0-1，再scale到0-2**(numBits-1)，再取整，再scale回去
    tensor=tensor.abs().mul(2**(numBits-1)).ceil().div(2**(numBits-1))
    训练过程中:forward时候调用；
    * (2)HCGS:
    在训练开始随机生成structure，训练过程中不变；
    mask,0
* 想法：
    * kaldi文档完整清楚，代码和配置都比较清楚；
    * CGS自定义变量和配置文件混乱，无文档，论文没写清楚，没必要完全复现
    * 可以仿照该量化方法和训练方法验证我们的结构化压缩方案，做Design Space Exploration
    
    
    