import torch
import torch.nn as nn
import torch.nn.functional as F

class net_D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(net_D, self).__init__()
        use_bias = norm_layer != nn.BatchNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        fcs=[nn.Linear(ndf * nf_mult*9, 1),
            nn.Sigmoid()]
        sequence += [nn.Conv2d(ndf * nf_mult,ndf * nf_mult, kernel_size=kw, stride=1, padding=0),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)] # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*fcs)

    def forward(self, input):
        """Standard forward."""
        out=self.model(input)
        print(out.size())
        out=out.view(-1,512*9)
        print(out.size())
        out=self.fc(out)
        return out
