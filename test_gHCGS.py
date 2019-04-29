import torch
import guided_hcgs

x = torch.randn(512,1944).cuda()

y = guided_hcgs.conn_mat(512,1944,[64,8,1],[25.0,75.0,25],x)
y = guided_hcgs.conn_mat(512,1944,[64,8,1],[25.0,75.0,25],x, mat_num='2')
print y
