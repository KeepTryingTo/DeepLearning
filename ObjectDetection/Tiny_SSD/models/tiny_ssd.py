"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/11/15-8:02
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
import torch.nn.functional as F


class FireM(nn.Module):
    def __init__(self,in_channels,out_channels,exp_ch = [49,53]):
        super().__init__()
        self.squueze = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                      kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        self.expand_conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=exp_ch[0],
                      kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(num_features=exp_ch[0]),
            nn.ReLU()
        )
        self.expand_conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=exp_ch[1],
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=exp_ch[1]),
            nn.ReLU()
        )

    def forward(self,x):
        sq = self.squueze(x)
        out = torch.cat([
            self.expand_conv1x1(sq),
            self.expand_conv3x3(sq)
        ],dim = 1)
        return out


class TinySSD(nn.Module):
    def __init__(self,num_classes = 21, anchor_nums=[6, 6, 6, 6, 6, 6]):
        super().__init__()
        self.num_classes = num_classes
        self.anchor_num = anchor_nums

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=57,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=57),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3,stride=2)
        )

        self.out_layers = [4,9,11,13]

        self.backbone = nn.Sequential(
            FireM(in_channels=57,out_channels=15,exp_ch=[49,53]), # fire1
            FireM(in_channels=102,out_channels=15,exp_ch=[54,52]), # fire2
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),

            FireM(in_channels=106, out_channels=29, exp_ch=[92, 94]), # fire3
            FireM(in_channels=186, out_channels=29, exp_ch=[90, 83]), # fire4
            nn.MaxPool2d(kernel_size=3, stride=2),

            FireM(in_channels=173, out_channels=44, exp_ch=[166, 161]), # fire5
            FireM(in_channels=327, out_channels=45, exp_ch=[155, 146]), # fire6
            FireM(in_channels=301, out_channels=49, exp_ch=[163, 171]), # fire7
            FireM(in_channels=334, out_channels=25, exp_ch=[29, 54]), # fire8
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),

            FireM(in_channels=83, out_channels=37, exp_ch=[45, 56]), # fire9
            nn.MaxPool2d(kernel_size=3, stride=2),

            FireM(in_channels=101, out_channels=38, exp_ch=[41, 44]), # fire10
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=85,out_channels=51,kernel_size=3,
                      stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=51,out_channels=46,kernel_size=3,
                      stride=2,padding=1),
            nn.ReLU()
        )

        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=46, out_channels=55, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=55, out_channels=85, kernel_size=2,
                      stride=1, padding=0),
            nn.ReLU()
        )

        self.loc_ch = [173,83,101,85,46,85]
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i, ch in enumerate(self.loc_ch):
            self.loc_layers.append(
                nn.Conv2d(in_channels=ch,out_channels=anchor_nums[i] * 4,
                          kernel_size=1,stride=1,padding=0)
            )
            self.cls_layers.append(
                nn.Conv2d(in_channels=ch, out_channels=anchor_nums[i] * num_classes,
                          kernel_size=1, stride=1, padding=0)
            )


    def forward(self,x):
        out = self.stem(x)

        sources = []
        for i,la in enumerate(self.backbone):
            out = la(out)
            if i in self.out_layers:
                sources.append(out)

        out = self.conv12(out)
        sources.append(out)

        out = self.conv13(out)
        sources.append(out)

        loc = []
        conf = []
        for i,(l,c) in enumerate(zip(self.loc_layers,self.cls_layers)):
            lo = l(sources[i])
            co = c(sources[i])
            loc.append(lo.view(sources[i].size()[0],-1,4))
            conf.append(co.view(sources[i].size()[0],-1,self.num_classes))

            """
            lo.shape: torch.Size([1, 24, 37, 37])
            lo.shape: torch.Size([1, 24, 18, 18])
            lo.shape: torch.Size([1, 24, 9, 9])
            lo.shape: torch.Size([1, 24, 4, 4])
            lo.shape: torch.Size([1, 24, 2, 2])
            lo.shape: torch.Size([1, 24, 1, 1])
            """
            # print('lo.shape: {}'.format(lo.size()))

        loc = torch.cat(loc,dim=1)
        conf = torch.cat(conf,dim=1)
        return loc,conf


def demo():
    x = torch.zeros(size=(1,3,300,300))
    model = TinySSD(num_classes=81)
    outs = model(x)
    for out in outs:
        print('out.shape: {}'.format(out.size()))

    # from torchinfo import summary
    # summary(model,input_size=(1,3,300,300))

if __name__ == '__main__':
    demo()
    pass