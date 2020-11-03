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
    训练过程中:forward时候调用
    * (2)HCGS:
    在训练开始随机生成structure，训练过程中不变；
    mask,0
* 想法：
    * kaldi文档完整清楚，代码和配置都比较清楚；
    * CGS自定义变量和配置文件混乱，无文档，论文没写清楚，没必要完全复现
    * 可以仿照该量化方法和训练方法验证我们的结构化压缩方案，做Design Space Exploration

### 4.20200909
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
    * (5):18.2% 8bweight_16bactivate+hcgs,补充：其实只有16/4*8/3=10.67x的LSTM压缩和8/4*16/12=2.67x的MLP压缩
    * (6):16.6% 8bweight_16bactivate
### 5.20200923
    //20200922 update
    原来的相当于lstm是4/16，5/8，对于MLP是4/8,12/16
    * (7):21.3% 8bit weight+16bit activate+ 16x LSTM hcgs+16x MLP hcgs,通过第一层4/16，第二层4/16的方式实现
            文件：0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200922_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs_16x
    * (7):21.2% 8bit weight+16bit activate+ 16x LSTM hcgs+16x MLP hcgs,通过第一层2/8，第二层4/16的方式实现
            文件：0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200922_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs_16x_v2
    * (7):21.6% 8bit weight+16bit activate+ 16x LSTM hcgs+16x MLP hcgs,通过第一层4/16，第二层2/8的方式实现
            文件：0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200922_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs_16x_v3
    * (7):22.1% 8bit weight+16bit activate+ 16x LSTM hcgs+16x MLP hcgs,通过第一层2/8，第二层2/8的方式实现
            文件：0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200922_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs_16x_v4
            
    * (8):26.1% 8bit weight+16bit activate+ 32x LSTM hcgs+32x MLP hcgs,通过第一层4/16，第二层２/16的方式实现
            文件：0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200922_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs_32x
    * (8):25.5% 8bit weight+16bit activate+ 32x LSTM hcgs+32x MLP hcgs,通过第一层２/16，第二层４/16的方式实现
            文件：0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200922_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs_32x_v2
    * (8):26.7% 8bit weight+16bit activate+ 32x LSTM hcgs+32x MLP hcgs,通过第一层1/8，第二层2/8的方式实现
            文件：0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200922_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs_32x_v3
    * (8):28.2% 8bit weight+16bit activate+ 32x LSTM hcgs+32x MLP hcgs,通过第一层2/8，第二层1/8的方式实现
            文件：0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200922_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs_32x_v4
            
    * (9):35.9% 8bit weight+16bit activate+ 64x LSTM hcgs+64x MLP hcgs,通过第一层２/16，第二层2/16的方式实现
            文件：0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200922_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs_64x
    * (9):35.9% 8bit weight+16bit activate+ 64x LSTM hcgs+64x MLP hcgs,通过第一层１/16，第二层４/16的方式实现
            文件：0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200922_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs_64x_v2
    * (9):35.5% 8bit weight+16bit activate+ 64x LSTM hcgs+64x MLP hcgs,通过第一层４/16，第二层１/16的方式实现
            文件：0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200922_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs_64x_v3
    * (9):39.7% 8bit weight+16bit activate+ 64x LSTM hcgs+64x MLP hcgs,通过第一层１/８，第二层１/８的方式实现
            文件：0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200922_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs_64x_v4
    * (9):36.4% 8bit weight+16bit activate+ 64x LSTM hcgs+64x MLP hcgs,通过第一层１/16，第二层２/８的方式实现
            文件：0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200922_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs_64x_v5
    * (9):37.0% 8bit weight+16bit activate+ 64x LSTM hcgs+64x MLP hcgs,通过第一层2/８，第二层１/16的方式实现
            文件：0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200922_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs_64x_v6
            
### 6.20200928
#### 6.1. Pattern的初步探索：
    1.初始化随机选择16个8*8/4的Pattern，每个512×512划分64*64个blocks随机选择pattern，训练过程中不变化。最终达到的准确度为21.3%，和同样为16x压缩的asu达到的效果相同。

### 7.20201103 Pruning and retrain
分两个任务做
* 训练一个pruning的网络，这样网络中就可以有mask，直接使用我们的无损编码，和别的无损编码相比较
问题1：网络中的pruning好像不是每个forward进行一次，而是每个epoch进行一次，为什么要这么做？
而且prune是在固定的epoch之后和固定的chunk之后才有，不是每个epoch，每个chunk都有的。
* 对于dense的网络
1.先测试精度
2.不同的compression rate进行pruning，测试精度，保存模型
3.pattern_search+pattern_pruning,输出overhead中间数据,输出真实的compression rate，测试精度，保存模型
4.按照该pattern retraining模型，每个epoch进行测试，保存精度。