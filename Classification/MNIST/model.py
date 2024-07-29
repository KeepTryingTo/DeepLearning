"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/7/28-20:33
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self,in_channels,num_classes = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,stride=1,padding=1,groups=8),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1)

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=8),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=8),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1)) #[b,128,h,w] => [b,128,1,1]

        self.classifier = nn.Sequential(
            nn.Linear(in_features=128,out_features=64),
            nn.Linear(in_features=64,out_features=num_classes)
        )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                #TODO fan_out 保留反向传播中权重方差的大小
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0,std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        out = self.stem(x)
        out = self.conv1(self.maxpool1(self.block1(out)))
        out = self.maxpool2(self.block2(out))
        out = self.maxpool3(self.block3(out))

        out = self.avgpool(out).view(x.size()[0],-1) # [b,128,1,1] => [b,128 * 1 * 1]
        out = self.classifier(out)
        out = nn.Softmax(dim=-1)(out)
        return out

def demo():
    model = Model(in_channels=1,num_classes=10)
    x = torch.rand(size=(2,1,28,28))
    out = model(x)
    print('out.shape: {}'.format(out.shape))

    from torchinfo import summary
    summary(model,input_size=(2,1,28,28))

if __name__ == '__main__':
    demo()
    pass


