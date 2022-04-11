from torch import nn
from torch import Tensor
from modules import *
from attention import *
import torch
import os
import math
from Res2Net_v1b import res2net50_v1b_26w_4s


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.d5=convbnrelu(2048,128)
        self.d4=convbnrelu(1024,128)
        self.d3=convbnrelu(512,128)
        self.d2=convbnrelu(256,128)
        self.d1=convbnrelu(64,128,k=3,p=1)
        self.sa5=SpatialAttention(k=3)
        self.ca4 =ChannelAttention(128, 8)

    def forward(self, F1,F2,F3,F4,F5):
     
        F1=self.d1(F1)
        F2=self.d2(F2)
        F3=self.d3(F3)
        F4=self.d4(F4)
        F5=self.d5(F5)

        F4=self.ca4(F4)
        F5=self.sa5(F5)

        return F1,F2,F3,F4,F5

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.c4 = convbnrelu(256, 128)
        self.c3 = convbnrelu(256, 128)
        self.c2 = convbnrelu(256, 128)
        self.c1 = convbnrelu(256, 64)

        self.srm3 = SRM(128)
        self.srm2 = SRM(128)
        self.srm1 = SRM(64)
        self.agg=convbnrelu(320,64)

    def forward(self, F1,F2,F3,F4,F5):

        P4 = torch.cat([F4, US2(F5)], dim=1)
        P4 = self.c4(P4)
        P3 = torch.cat([F3, US2(P4)], dim=1)
        P3 = self.srm3(self.c3(P3))
        P2 = torch.cat([F2, US2(P3)], dim=1)
        P2 = self.srm2(self.c2(P2))
        P1 = torch.cat([F1, US2(P2)], dim=1)
        P1 = self.srm1(self.c1(P1))

        S=torch.cat([P1,US2(P2),US4(P3)],dim=1)
        S=self.agg(S)

        return S,P4

class Decoder0(nn.Module):
    def __init__(self):
        super(Decoder0, self).__init__()
        self.d5=convbnrelu(2048,128)
        self.d4=convbnrelu(1024,128)
        self.d3=convbnrelu(512,128)
        self.d2=convbnrelu(256,128)

        self.c4=convbnrelu(256,128)
        self.c3=convbnrelu(256,128)
        self.c2=convbnrelu(256,128)
        self.c1=convbnrelu(192,64)

    def forward(self, F1,F2,F3,F4,F5):
        F5=self.d5(F5)
        F4=self.d4(F4)
        F3=self.d3(F3)
        F2=self.d2(F2)

        P4 = torch.cat([F4, US2(F5)], dim=1)
        P4 = self.c4(P4)
        P3 = torch.cat([F3, US2(P4)], dim=1)
        P3 = self.c3(P3)
        P2 = torch.cat([F2, US2(P3)], dim=1)
        P2 = self.c2(P2)
        P1 = torch.cat([F1, US2(P2)], dim=1)
        S = self.c1(P1)

        return S


class AGNet(nn.Module):
    def __init__(self):

        super(AGNet, self).__init__()
        self.bkbone = res2net50_v1b_26w_4s(pretrained=True)
        self.encoder=Encoder()
        self.decoder = Decoder()

        self.head = nn.ModuleList([])
        for i in range(2):
            self.head.append(SalHead(64,3))

    def forward(self, x):
        x = self.bkbone.conv1(x)
        x = self.bkbone.bn1(x)
        x0 = self.bkbone.relu(x)
        x = self.bkbone.maxpool(x0)
        # ---- low-level features ----
        x1 = self.bkbone.layer1(x)
        x2 = self.bkbone.layer2(x1)
        x3 = self.bkbone.layer3(x2)
        x4 = self.bkbone.layer4(x3)

        f1,f2,f3,f4,f5=self.encoder(x0,x1,x2,x3,x4)
        S,P4= self.decoder(f1,f2,f3,f4,f5)

        sm = self.head[0](US2(S))
        se = self.head[1](US2(S))

        return sm,se








