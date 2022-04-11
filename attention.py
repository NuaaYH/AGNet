from AGNet.utils import *
from AGNet.modules import *
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self,k=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=k, padding=k//2, bias=False) # infer a one-channel attention map
        self.la = nn.Sequential(nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=False), nn.Sigmoid())
    def forward(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = torch.mean(ftr, dim=1, keepdim=True) # [B, 1, H, W], average
        ftr_max, _ = torch.max(ftr, dim=1, keepdim=True) # [B, 1, H, W], max
        ftr_cat = torch.cat([ftr_avg, ftr_max], dim=1) # [B, 2, H, W]
        att_map = F.sigmoid(self.conv(ftr_cat)) # [B, 1, H, W]
        return att_map*ftr+self.la(ftr)*ftr


class ChannelAttention(nn.Module):
    def __init__(self, in_planes,g):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes, 1,groups=g, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1,groups=g, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ym=self.max_pool(x)
        ya=self.avg_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(ya)))
        max_out = self.fc2(self.relu1(self.fc1(ym)))
        out = self.sigmoid(avg_out+max_out)
        return out*x

