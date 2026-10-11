import os
import cv2
import torch
import numpy as np
from torch import nn
from torchstat import stat
from  torchvision import transforms
from torchvision.models import ResNet
from torch.utils.data import Dataset,DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class myBuildModel_1(nn.Module):
    def __init__(self):
        super(myBuildModel_1, self).__init__()
        self.PreTrainingModel=fasterrcnn_resnet50_fpn(pretrained=True,progress=True,num_classes=4)

    def forward(self,x_images,x_targets):
        out=self.PreTrainingModel(x_images,x_targets)
        return out

class myBuildModel_2(nn.Module):
    def __init__(self):
        super(myBuildModel_2, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        # #最大平均池化
        # self.avgpool=nn.AdaptiveMaxPool2d(output_size=(1,1))
        # #展平
        # self.flatten=nn.Flatten()
        #
        # #输出坐标值
        # self.positions=nn.Linear(in_features=512,out_features=4,bias=True)
        # #输出类别+1(background)
        # self.labels=nn.Linear(in_features=512,out_features=3,bias=True)

        self.position=nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2,2), padding=2),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=4, kernel_size=(3, 3), stride=(1,1), padding=0),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(inplace=True)
        )

        self.label=nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=2),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        out=self.layer1(x)
        # out=self.avgpool(out)
        # out=self.flatten(out)
        # outP=self.positions(out)
        # outL=self.labels(out)
        # return outP,outL
        outP=self.position(out)
        outP=torch.squeeze(input=outP,dim=2)
        outP=torch.squeeze(input=outP,dim=2)

        outL = self.label(out)
        outL = torch.squeeze(input=outL, dim=2)
        outL = torch.squeeze(input=outL, dim=2)
        return outP,outL

#模仿VGG16搭建的模型（其中一些地方已经精剪）
class myVGG16(nn.Module):
    def __init__(self):
        super(myVGG16, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=256),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=512),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.position = nn.Sequential(
            #首先使用1X1卷积减少通道数，减少参数量
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1,1), padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            # 首先使用1X1卷积减少通道数，减少参数量
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),


            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(inplace=True)
        )

        self.label = nn.Sequential(
            # 首先使用1X1卷积减少通道数，减少参数量
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            # 首先使用1X1卷积减少通道数，减少参数量
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        out=self.layer1(x)
        outP = self.position(out)
        outP = torch.squeeze(input=outP, dim=2)
        outP = torch.squeeze(input=outP, dim=2)

        outL = self.label(out)
        outL = torch.squeeze(input=outL, dim=2)
        outL = torch.squeeze(input=outL, dim=2)
        return outP, outL

model=myVGG16()
x=torch.randn(size=(3,3,224,224))
print(model)
outP,outL=model(x)
print('P.shape: {}'.format(outP.shape))
print('L.shape: {}'.format(outL.shape))

print('modelParameters: {}'.format(stat(model,(3,224,224))))
out=model(x)
# print(out.shape)


if __name__ == '__main__':
    # model=myBuildModel()
    # model = myBuildModel_2()
    # x = torch.randn(size=(3, 3, 224, 224))
    # outP, outL = model(x)
    # print('P.shape: {}'.format(outP.shape))
    # print('L.shape: {}'.format(outL.shape))
    # print('model.shape: {}'.format(model(x).shape))
    pass
