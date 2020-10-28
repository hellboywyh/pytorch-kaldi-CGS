<!--
 * @Description: 
 * @version: 
 * @Author: Wang Yanhong
 * @email: 284520535@qq.com
 * @Date: 2020-10-21 00:40:52
 * @LastEditors: Wang Yanhong
 * @LastEditTime: 2020-10-27 07:47:39
-->
# 测试基准
* 模型
cfg文件：cfg/20200910_QuantExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_wohcgs_v1.cfg
exp文件：exp/20200910_QuantExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_wohcgs_v1
without pruning 

* 运行命令
`python run_test.py cfg/20201021_Pattern_Search/TIMIT_LSTM_fmllr_L2_8bw_16ba_wohcgs_v1.cfg`

* 基准表现
WER:16.7%

* Debug
data_chunks-----(经过test)-----exp_files/*.arks
exp_files/*.arks-----(经过decoder)-----decode...dnn2/score

```python
#单独decoder命令：
kaldi_decoding_scripts//decode_dnn.sh /yhwang/0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20201021_Pattern_Search/TIMIT_LSTM_fmllr_L2_8bw_16ba_16_8x8x_16/decoding_TIMIT_test_out_dnn2.conf /yhwang/0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20201021_Pattern_Search/TIMIT_LSTM_fmllr_L2_8bw_16ba_16_8x8x_16/decode_TIMIT_test_out_dnn2 /yhwang/0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20201021_Pattern_Search/TIMIT_LSTM_fmllr_L2_8bw_16ba_16_8x8x_16/exp_files/forward_TIMIT_test_ep*_ck*_out_dnn2_to_decode.ark

kaldi_decoding_scripts//decode_dnn.sh /yhwang/0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20201021_Pattern_Search/TIMIT_LSTM_fmllr_L2_8bw_16ba_wohcgs_v1/decoding_TIMIT_test_out_dnn2.conf /yhwang/0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20201021_Pattern_Search/TIMIT_LSTM_fmllr_L2_8bw_16ba_wohcgs_v1/decode_TIMIT_test_out_dnn2 /yhwang/0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20201021_Pattern_Search/TIMIT_LSTM_fmllr_L2_8bw_16ba_wohcgs_v1/exp_files/forward_TIMIT_test_ep*_ck*_out_dnn2_to_decode.ark

#失败的输出：
run.pl: 10 / 10 failed, log is in /yhwang/0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20201021_Pattern_Search/TIMIT_LSTM_fmllr_L2_8bw_16ba_16_8x8x_16/decode_TIMIT_test_out_dnn2/scoring/log/score.*.log


#输出结果命令：
./check_res_dec.sh /yhwang/0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20201021_Pattern_Search/TIMIT_LSTM_fmllr_L2_8bw_16ba_16_8x8x_16/decode_TIMIT_test_out_dnn2

./check_res_dec.sh /yhwang/0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20200910_QuanExp/TIMIT_LSTM_fmllr_L2_8bw_16ba_whcgs/decode_TIMIT_test_out_dnn2
#成功的输出
%WER 78.8 | 192 7215 | 21.2 1.7 77.1 0.0 78.8 100.0 | -3.495 | /yhwang/0-Projects/0-kaldi-lstm/2-pytorch-kaldi-cgs/exp/20201021_Pattern_Search/TIMIT_LSTM_fmllr_L2_8bw_16ba_16_8x8x_8/decode_TIMIT_test_out_dnn2/score_1/ctm_39phn.filt.sys
#失败不输出
```

1.decoding.py 没有问题，只要生成的ark正确就能正确运行


