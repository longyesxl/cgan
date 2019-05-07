import torch
import torch.nn as nn
class gen_net(nn.Module):
    def __init__(self,inner_nc,ndf=64,cn=8):
        super(gen_net, self).__init__()
        uprelu1 = nn.LeakyReLU(0.2, True)
        upconv1 = nn.ConvTranspose2d(inner_nc, ndf,kernel_size=4, stride=2,padding=1)
        upnorm1 = nn.BatchNorm2d(ndf)
        uprelu2 = nn.LeakyReLU(0.2, True)
        upconv2 = nn.ConvTranspose2d(ndf, ndf*2,kernel_size=4, stride=2,padding=1)
        upnorm2 = nn.BatchNorm2d(ndf*2)
        uprelu3 = nn.LeakyReLU(0.2, True)
        upconv3 = nn.ConvTranspose2d(ndf*2, ndf*4,kernel_size=4, stride=2,padding=1)
        upnorm3 = nn.BatchNorm2d(ndf*4)
        uprelu4 = nn.LeakyReLU(0.2, True)
        upconv4 = nn.ConvTranspose2d(ndf*4, ndf*8,kernel_size=4, stride=2,padding=1)
        upnorm4 = nn.BatchNorm2d(ndf*8)
        upmodel=[upconv1,upnorm1,uprelu1,upconv2,upnorm2,uprelu2,upconv3,upnorm3,uprelu3,upconv4,upnorm4,uprelu4]
        for i in range(cn-5):
            upconv = nn.ConvTranspose2d(ndf*8, ndf*8,kernel_size=4, stride=2,padding=1)
            upnorm = nn.BatchNorm2d(ndf*8)
            uprelu = nn.LeakyReLU(0.2, True)
            upmodel+=[upconv,upnorm,uprelu]
        upconvl = nn.ConvTranspose2d(ndf*8, 1,kernel_size=4, stride=2,padding=1)
        upmodel+=[upconvl,nn.Tanh()]
        self.model = nn.Sequential(*upmodel)
    def forward(self, x):
        return self.model(x)
        
