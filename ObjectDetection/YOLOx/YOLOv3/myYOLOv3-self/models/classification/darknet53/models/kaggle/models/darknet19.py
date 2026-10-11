"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/23 20:26
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Variable


class Darknet19(nn.Module):
    # (64,1) means conv kernel size is 1, by default is 3.
    cfg1 = [32, 'M', 64, 'M', 128, (64,1), 128, 'M', 256, (128,1), 256, 'M', 512, (256,1), 512, (256,1), 512]  # conv1 - conv13
    cfg2 = ['M', 1024, (512,1), 1024, (512,1), 1024]  # conv14 - conv18

    def __init__(self,num_classes = 20):
        super(Darknet19, self).__init__()
        self.num_classes = num_classes
        self.layer1 = self._make_layers(self.cfg1, in_planes=3)
        self.layer2 = self._make_layers(self.cfg2, in_planes=512)

        #passthrough layer
        self.passlayer = nn.Sequential(
            nn.Conv2d(512,64,kernel_size=(1,1),stride=(1,1),padding=0),
            nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        )
        #### Add new layers
        self.conv19 = nn.Conv2d(1024, 1024, kernel_size=(3,3), stride=(1,1), padding=1)
        self.bn19 = nn.BatchNorm2d(1024)

        self.conv20 = nn.Conv2d(1024, 512,kernel_size=(3,3), stride=(1,1), padding=1)
        self.bn20 = nn.BatchNorm2d(512)
        # Currently I removed the passthrough layer for simplicity
        self.conv21 = nn.Conv2d(768, 512, kernel_size=(3,3), stride=(1,1), padding=1)
        self.bn21 = nn.BatchNorm2d(512)
        # Outputs: 5boxes * (4coordinates + 1confidence + 20classes)
        self.conv22 = nn.Conv2d(512, 5*(5+self.num_classes), kernel_size=(3,3), stride=(1,1), padding=1)

        self.avgpooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(in_features=5*(5+self.num_classes),out_features=self.num_classes)

    def _make_layers(self, cfg, in_planes):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                out_planes = x[0] if isinstance(x, tuple) else x
                ksize = x[1] if isinstance(x, tuple) else 3
                layers += [nn.Conv2d(in_planes, out_planes, kernel_size=ksize, padding=(ksize-1)//2),
                           nn.BatchNorm2d(out_planes),
                           nn.LeakyReLU(0.1, True)]
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        b,c,h,w = x.size()
        out = self.layer1(x)
        passlayer = self.passlayer(out)
        out = self.layer2(out)
        out = F.leaky_relu(self.bn19(self.conv19(out)), 0.1)
        out = F.leaky_relu(self.bn20(self.conv20(out)), 0.1)
        #passthrough layer
        out = torch.cat([out,passlayer],dim = 1)
        out = F.leaky_relu(self.bn21(self.conv21(out)), 0.1)
        out = self.conv22(out)
        avgpool = self.avgpooling(out).view(b,-1)
        out = self.classifier(avgpool)
        return out


def demo():
    net = Darknet19(num_classes=1000)
    x = torch.zeros(size = (1,3,224,224))
    out = net(x)
    print('out.shape: {}'.format(out.shape))  # [1,125,13,13]

    from torchinfo import summary
    summary(net,input_size=(1,3,224,224))

if __name__ == '__main__':
    demo()
    pass
