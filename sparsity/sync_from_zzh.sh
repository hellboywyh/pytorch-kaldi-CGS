set -x -e -u -o pipefail
cd ../../4-ZZH-STT-wavenet/speech-to-text-wavenet/
git pull
cp torch_lyuan/sparsity.py ../../2-pytorch-kaldi-cgs/sparsity/
