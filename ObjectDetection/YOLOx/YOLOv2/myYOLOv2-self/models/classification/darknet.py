"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/10 21:54
"""

import torch
from torch import nn
import torch.nn.functional as F
from models.modules import *

class DarkNet(nn.Module):
    def __init__(
            self,in_channels = 3,img_size = 448,num_classes = 1000,
            channels_list = (64,192,256,512,1024)
    ):
        super(DarkNet, self).__init__()
        self.stem = nn.Sequential(
            ConvBNRE(in_channels=3,out_channels=channels_list[0],kernel_size=7,stride=2,padding=3),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),

            ConvBNRE(in_channels=channels_list[0],out_channels=channels_list[1],kernel_size=3,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.body_conv = nn.Sequential(
            ConvBNRE(in_channels=channels_list[1],out_channels=channels_list[2],kernel_size=1,stride=1),
            ConvBNRE(in_channels=channels_list[2],out_channels=channels_list[3],kernel_size=3,stride=1,padding=1),
            ConvBNRE(in_channels=channels_list[3],out_channels=channels_list[3],kernel_size=1,stride=1),
            ConvBNRE(in_channels=channels_list[3],out_channels=channels_list[3],kernel_size=3,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),

            ConvBNRE(in_channels=channels_list[3],out_channels=channels_list[3],kernel_size=1,stride=1),
            ConvBNRE(in_channels=channels_list[3],out_channels=channels_list[3],kernel_size=3,stride=1,padding=1),
            ConvBNRE(in_channels=channels_list[3],out_channels=channels_list[3],kernel_size=3,stride=1,padding=1),
            ConvBNRE(in_channels=channels_list[3],out_channels=channels_list[3],kernel_size=3,stride=2,padding=1),

            ConvBNRE(in_channels=channels_list[3],out_channels=channels_list[3],kernel_size=3,stride=1,padding=1),
            ConvBNRE(in_channels=channels_list[3],out_channels=channels_list[4],kernel_size=3,stride=1,padding=1)
        )
        self.globalAvg = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.final_conv = nn.Sequential(
            nn.Linear(in_features=1024,out_features=2048),
            nn.Linear(in_features=2048,out_features=num_classes)
        )

    def forward(self,x):
        b,c,h,w = x.size()
        stem = self.stem(x)
        backbone = self.body_conv(stem)
        globalAvg = self.globalAvg(backbone).view(b,-1)
        out = self.final_conv(globalAvg)
        out = out.view(b,7,7,30)
        return out

def demoDarkNet():
    model = DarkNet(in_channels=3,img_size=448)
    x = torch.zeros(size = (1,3,448,448))
    out = model(x)
    print('out.shape: {}'.format(out.shape))

    from torchinfo import summary
    summary(model,input_size=(1,3,448,448))

if __name__ == '__main__':
    demoDarkNet()
    pass