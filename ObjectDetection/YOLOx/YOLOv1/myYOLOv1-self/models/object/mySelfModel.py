"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/9/23 10:45
"""

#######################
# name: EDANet full model definition reproduced by Pytorch(v0.4.1)
# data: Sept 2018
# author:PengfeiWang(pfw813@gmail.com)
# paper: Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation
#######################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#定义输出网格的大小以及B
# S = 7
# B = 2
# #对应VOC类别数
# C = 20


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super(DownsamplerBlock, self).__init__()

        self.ninput = ninput
        self.noutput = noutput

        if self.ninput < self.noutput:
            # Wout > Win
            self.conv = nn.Conv2d(ninput, noutput - ninput, kernel_size=3, stride=2, padding=1)
            self.pool = nn.MaxPool2d(2, stride=2)
        else:
            # Wout < Win
            self.conv = nn.Conv2d(ninput, noutput, kernel_size=3, stride=2, padding=1)

        self.bn = nn.BatchNorm2d(noutput)

    def forward(self, x):
        if self.ninput < self.noutput:
            output = torch.cat([self.conv(x), self.pool(x)], 1)
        else:
            output = self.conv(x)

        output = self.bn(output)
        return F.relu(output)


class EDABlock(nn.Module):
    def __init__(self, ninput, dilated, k=40, dropprob=0.02):
        super(EDABlock, self).__init__()

        # k: growthrate
        # dropprob:a dropout layer between the last ReLU and the concatenation of each module

        self.conv1x1 = nn.Conv2d(in_channels=ninput, out_channels=k, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(k)

        self.conv3x1_1 = nn.Conv2d(k, k, kernel_size=(3, 1), padding=(1, 0))
        self.conv1x3_1 = nn.Conv2d(k, k, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(k)

        self.conv3x1_2 = nn.Conv2d(k, k, (3, 1), stride=1, padding=(dilated, 0), dilation=dilated)
        self.conv1x3_2 = nn.Conv2d(k, k, (1, 3), stride=1, padding=(0, dilated), dilation=dilated)
        self.bn2 = nn.BatchNorm2d(k)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        input = x

        output = self.conv1x1(x)
        output = self.bn0(output)
        output = F.relu(output)

        output = self.conv3x1_1(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        output = F.relu(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        output = torch.cat([output, input], 1)
        # print output.size() #check the output
        return output


class CBR(nn.Module):
    def __init__(self,in_planes,out_planes,kSize = 3,stride = 1,padding = 1):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size = (kSize,kSize),
            stride=(stride,stride),
            padding = padding,
            dilation = (padding,padding)
        )
        self.BN = nn.BatchNorm2d(
            num_features=in_planes
        )
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.relu(self.BN(self.conv(x)))
        return out


class EDANet(nn.Module):
    def __init__(self, num_classes=20,B = 2,S = 7):
        super(EDANet, self).__init__()

        self.S = S
        self.B = B
        self.num_classes = num_classes

        self.layers = nn.ModuleList()
        self.dilation1 = [1, 1, 1, 2, 2]
        self.dilation2 = [2, 2, 4, 4, 8, 8, 16, 16]

        # DownsamplerBlock1
        self.layers.append(DownsamplerBlock(3, 15))

        # DownsamplerBlock2
        self.layers.append(DownsamplerBlock(15, 60))

        # EDA module 1-1 ~ 1-5
        for i in range(5):
            self.layers.append(EDABlock(60 + 40 * i, self.dilation1[i]))

        # DownsamplerBlock3
        self.layers.append(DownsamplerBlock(260, 130))

        # EDA module 2-1 ~ 2-8
        for j in range(8):
            self.layers.append(EDABlock(130 + 40 * j, self.dilation2[j]))

        # Projection layer
        self.project_layer = nn.Conv2d(450, 512, kernel_size=1)

        #final con
        self.final_conv = nn.ModuleList()
        in_planes = 512
        for i in range(3):
            self.final_conv.append(
                CBR(in_planes=in_planes,out_planes=512,kSize=3,stride=2,padding=1)
            )
            in_planes = 512

        #output head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(),
            nn.Linear(in_features=512,out_features=256),
            nn.Dropout(0.1),
            nn.Linear(in_features=256,out_features=self.S * self.S * (self.num_classes + self.B * 5))
        )

        self.weights_init()

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        b,c,h,w = x.size()
        output = x

        for layer in self.layers:
            output = layer(output)

        output = self.project_layer(output)

        for i,conv in enumerate(self.final_conv):
            output = conv(output)

        output = self.head(output).view(b,self.S,self.S,self.num_classes + self.B * 5)
        output = torch.sigmoid(output)
        return output


def demo():
    model = EDANet(
        num_classes=20
    )
    x = torch.zeros(size=(1, 3, 448, 448))
    model.train()
    out = model(x)
    print('training.out.shape: {}'.format(out.shape))

    model.eval()
    out = model(x)
    print('inference.out.shape: {}'.format(out.shape))

    from torchinfo import summary
    summary(model,input_size=(1,3,512,512))


if __name__ == '__main__':
    demo()
    pass

