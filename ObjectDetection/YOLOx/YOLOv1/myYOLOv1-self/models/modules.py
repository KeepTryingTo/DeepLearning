"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/2 10:02
"""
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups = 1,
            same = False
    ):
        super(ConvBlock, self).__init__()
        if same and isinstance(kernel_size,list) or isinstance(kernel_size,tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation,(kernel_size[1] - 1) // 2 * dilation)
        elif same and isinstance(kernel_size,int):
            padding = ((kernel_size - 1) // 2 * dilation,(kernel_size - 1) // 2 * dilation)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                stride=stride,padding=padding,dilation=dilation,groups=groups
            )
        )
    def forward(self,x):
        return self.conv(x)

class ConvBNRE(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            same=False
    ):
        super(ConvBNRE, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                      stride=stride,padding=padding,groups=groups,dilation=dilation,same=same),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self,x):
        return self.conv(x)


class Laplace(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            groups = 1
    ):
        super(Laplace, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel = torch.tensor(
            [[1,1,1],[1,-8,1],[1,1,1]],
            dtype=torch.float,requires_grad=False
        )

    def forward(self,x):
        #weight: [out_channels,in_channels / groups,kH,kW]
        kernel = self.kernel.repeat(self.out_channels,self.in_channels // self.groups,1,1)
        # print(kernel.shape)
        laplace = F.conv2d(
            x,weight=kernel,
            stride=1,padding=(1,1),groups=self.groups,dilation=1
        )
        return laplace

def demoLaplace():
    from PIL import Image
    operatorLaplace = Laplace(in_channels=3,out_channels=3)
    img = cv2.imread(r'../images/1.jpg',flags=1)
    img = cv2.resize(src=img,dsize=(800,600))
    cv2.imshow('original',img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose([2,0,1])).unsqueeze(0).float()
    out = operatorLaplace(img).squeeze(0).numpy().transpose([1,2,0])
    print('out.shape: {}'.format(np.shape(out)))
    cv2.imshow('laplace',out)
    cv2.waitKey(0)

if __name__ == '__main__':
    demoLaplace()
    pass