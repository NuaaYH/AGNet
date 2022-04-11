from utils import *
import torch.nn as nn
import torch
import math
from einops import rearrange, repeat


class ConvBlock(nn.Module):
    def __init__(self, ic, oc, ks, use_bn, nl):
        # ic: input channels
        # oc: output channels
        # ks: kernel size
        # use_bn: True or False
        # nl: type of non-linearity, 'Non' or 'ReLU' or 'Sigmoid'
        super(ConvBlock, self).__init__()
        assert ks in [1, 3, 5, 7]
        assert isinstance(use_bn, bool)
        assert nl in ['None', 'ReLU', 'Sigmoid']
        self.use_bn = use_bn
        self.nl = nl
        if ks == 1:
            self.conv = nn.Conv2d(ic, oc, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(ic, oc, kernel_size=ks, padding=(ks-1)//2, bias=False)
        if self.use_bn == True:
            self.bn = nn.BatchNorm2d(oc)
        if self.nl == 'ReLU':
            self.ac = nn.ReLU(inplace=True)
        if self.nl == 'Sigmoid':
            self.ac = nn.Sigmoid()
    def forward(self, x):
        y = self.conv(x)
        if self.use_bn == True:
            y = self.bn(y)
        if self.nl != 'None':
            y = self.ac(y)
        return y


class SalHead(nn.Module):
    def __init__(self, in_channels, inter_ks):
        super(SalHead, self).__init__()
        self.conv_1 = ConvBlock(in_channels, in_channels//2, inter_ks, True, 'ReLU')
        self.conv_2 = ConvBlock(in_channels//2, in_channels//2, 3, True, 'ReLU')
        self.conv_3 = ConvBlock(in_channels//2, in_channels//8, 3, True, 'ReLU')
        self.conv_4 = ConvBlock(in_channels//8, 1, 1, False, 'Sigmoid')

    def forward(self, dec_ftr):
        dec_ftr_ups = dec_ftr#US2(dec_ftr)
        outputs = self.conv_4(self.conv_3(self.conv_2(self.conv_1(dec_ftr_ups))))
        return outputs


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=1, s=1, p=0, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)


class AsyConv(nn.Module):
    def __init__(self, inc,outc,k,p):
        super(AsyConv, self).__init__()
        self.inc=inc
        self.outc=outc
        self.c=nn.Sequential(
            nn.Conv2d(inc, inc, (1, k), padding=0, bias=False),
            nn.Conv2d(inc, outc, (k, 1), padding=p, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out=self.c(x)
        return out



class SRM(nn.Module):
    def __init__(self, c):
        super(SRM, self).__init__()
        self.conv = nn.Conv2d(in_channels=c, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.c1 = convbnrelu(c,c)
        self.c2 = convbnrelu(c,c)
        self.c3 = convbnrelu(c,c)

    def forward(self, x):
        a=torch.sigmoid(self.conv(x))
        ra=1-a
        x1=self.c1(a*x)
        x2=self.c2(ra*x)
        out=self.c3(x1+x2)
        return out


class AGG(nn.Module):
    def __init__(self,c1,c2):
        super(AGG, self).__init__()
        self.ca=nn.Conv2d(2*c1+c2,c2,kernel_size=1,padding=0,bias=False)
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.agg=convbnrelu(2*c1+c2,c2)
    def forward(self, x1,x2,x3):
        out=torch.cat([x1,x2,x3],dim=1)
        att=self.ca(self.avg(out))
        out=self.agg(out)*torch.sigmoid(att)
        return out



















