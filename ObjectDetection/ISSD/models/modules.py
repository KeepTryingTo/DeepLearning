"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/11/2-13:46
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import torch
from torch import nn
import torch.nn.functional as F


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self,in_channels):
        super().__init__()

        self.out_channels = in_channels
        self.branch1 = nn.Sequential(
            BasicConv(in_planes=in_channels,out_planes=in_channels,kernel_size=1,
                      stride=1,relu=False),
        )

        self.branch2 = nn.Sequential(
            BasicConv(in_planes=in_channels,out_planes=in_channels,kernel_size=1,
                      relu=False),
            BasicConv(in_planes=in_channels, out_planes=in_channels, kernel_size=3,
                      stride=1,padding=1,relu=False)
        )

        self.branch3 = nn.Sequential(
            BasicConv(in_planes=in_channels, out_planes=in_channels, kernel_size=1,
                      relu=False),
            BasicConv(in_planes=in_channels, out_planes=in_channels, kernel_size=3,
                      stride=1, padding=1, relu=False),
            BasicConv(in_planes=in_channels, out_planes=in_channels, kernel_size=3,
                      stride=1, padding=1, relu=False)
        )

        self.final = nn.Sequential(
            BasicConv(in_planes=in_channels * 3, out_planes=in_channels, kernel_size=1,
                      bn=False,relu=False),
            BasicConv(in_planes=in_channels, out_planes=in_channels, kernel_size=1,
                      bn=False,relu=False)
        )

    def forward(self,x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1,out2,out3],dim=1)
        out = self.final(out) + x
        return out


if __name__ == '__main__':
    x = torch.zeros(size=(1,32,128,128))
    model = InceptionModule(in_channels=32)
    out = model(x)
    print('out.shape: {}'.format(out.size()))
    pass
