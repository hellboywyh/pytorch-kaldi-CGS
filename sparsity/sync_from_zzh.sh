###
 # @Author: your name
 # @Date: 2020-11-03 12:01:30
 # @LastEditTime: 2020-11-05 08:21:41
 # @LastEditors: Wang Yanhong
 # @Description: In User Settings Edit
 # @FilePath: /2-pytorch-kaldi-cgs/sparsity/sync_from_zzh.sh
### 
set -x -e -u -o pipefail
cd ../../4-ZZH-STT-wavenet/speech-to-text-wavenet/
cp ../../2-pytorch-kaldi-cgs/sparsity/sparsity.py torch_lyuan/sparsity.py 
cp ../../2-pytorch-kaldi-cgs/sparsity/write_excel.py torch_lyuan/write_excel.py 
git stash
git pull
git stash pop
git commit -am "WYH Update"
git push
cp torch_lyuan/sparsity.py ../../2-pytorch-kaldi-cgs/sparsity/  
cp torch_lyuan/write_excel.py ../../2-pytorch-kaldi-cgs/sparsity/