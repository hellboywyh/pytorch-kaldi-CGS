import torch
import guided_hcgs
import hcgs

x = torch.randn(512,440).cuda()

y = guided_hcgs.conn_mat(512,440,[64,4],[25.0,75.0],x,for_test=True)
y = hcgs.conn_mat(512,440,[64,4],[25.0,75.0], mat_num='2',for_test=True)
# print y
