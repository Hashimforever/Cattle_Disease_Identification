import torch
import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_out = self.sigmoid(avg_out + max_out)
        return channel_out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_weighted = torch.mean(x, dim=1, keepdim=True)
        max_weighted, _ = torch.max(x, dim=1, keepdim=True)
        spatial_descriptor = torch.cat([avg_weighted, max_weighted], dim=1)
        spatial_out = self.conv(spatial_descriptor)
        spatial_out = self.sigmoid(spatial_out)
        return spatial_out

class CCSAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CCSAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x