from test import LSTM
import torch

model = LSTM(20)
print
x = torch.randn(20,20,20)#.cuda()
y = model.prune()
print y