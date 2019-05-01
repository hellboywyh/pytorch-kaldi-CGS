import torch
import guided_hcgs

x = torch.randn(512,1944).cuda()

y = guided_hcgs.conn_mat(512,1944,[73,13,4],[25.0,75.0,25],x,for_test=True)
y = guided_hcgs.conn_mat(512,1944,[73,13,4],[25.0,75.0,25],x, mat_num='2',for_test=True)
print y
