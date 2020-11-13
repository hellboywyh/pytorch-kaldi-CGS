###
 # @Author: your name
 # @Date: 2020-11-03 12:01:30
 # @LastEditTime: 2020-11-10 07:06:03
 # @LastEditors: Wang Yanhong
 # @Description: In User Settings Edit
 # @FilePath: /2-pytorch-kaldi-cgs/sparsity/sync_from_zzh.sh
### 
set -x -e -u -o pipefail
cd ../../4-ZZH-STT-wavenet/speech-to-text-wavenet/
git pull
cp torch_lyuan/sparsity.py ../../2-pytorch-kaldi-cgs/sparsity/  
cp torch_lyuan/write_excel.py ../../2-pytorch-kaldi-cgs/sparsity/


# cd ../../4-ZZH-STT-wavenet/speech-to-text-wavenet/
# cp ../../2-pytorch-kaldi-cgs/sparsity/sparsity.py torch_lyuan/sparsity.py 
# cp ../../2-pytorch-kaldi-cgs/sparsity/write_excel.py torch_lyuan/write_excel.py 

# git stash
# git pull
# git stash pop
# git commit -am "add mask.squeeze"
# git push