# Author: Zylo117

import math

import torch
from torch import nn
import torch.nn.functional as F


#TODO 输入输出的特征图大小保持不变
class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            bias=bias, groups=groups
        )
        self.stride = self.conv.stride #TODO （1,1）
        self.kernel_size = self.conv.kernel_size #TODO （3,3）
        self.dilation = self.conv.dilation

        #TODO 因此这里的过程都不会执行
        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        #TODO 计算需要在边缘填充的额外高度和宽度
        extra_h = (
                    math.ceil(w / self.stride[1]) - 1
                  ) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (
                    math.ceil(h / self.stride[0]) - 1
                  ) * self.stride[0] - h + self.kernel_size[0]

        #TODO 计算上下，左右填充的大小
        left = extra_h // 2
        right = extra_h - left

        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x

def demoConv2dStaticSamePadding():
    model = Conv2dStaticSamePadding(
        in_channels=3,
        out_channels=32,
        kernel_size=3,
    )

    x = torch.zeros(size=(1,3,512,512))
    out = model(x)
    print('out.shape: {}'.format(out.shape))

demoConv2dStaticSamePadding()

class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x

def demoMaxPool2dStaticSamePadding():
    model = MaxPool2dStaticSamePadding(
        kernel_size = 3,
        stride = 2
    )

    x = torch.zeros(size=(1,3,512,512))
    out = model(x)
    print('out.shape: {}'.format(out.shape))

# demoMaxPool2dStaticSamePadding()
