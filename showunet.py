import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from gen_net import gen_net
from net_D import net_D
from torchsummary import summary

print(torch.cuda.is_available())
input_arr=torch.rand(1,100,1,1).cuda()
#net = net_D(3,64,n_layers=5).cuda()
net = gen_net(100).cuda()
summary(net, (100,1,1))
with SummaryWriter(comment='net') as w:
    w.add_graph(net,(input_arr,))