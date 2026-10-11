"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/14 15:32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from collections import OrderedDict


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))

#################################
#       Transfer Learning       #
#################################
class YOLOv2ResNet(nn.Module):
    def __init__(self,B = 2,S = 7,C = 20):
        super().__init__()
        self.depth = B * 5 + C

        # Load backbone ResNet
        backbone = resnet50(pretrained=True, progress=True)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        # backbone.requires_grad_(False)  # Freeze backbone weights
        self.model = nn.Sequential(
            DetectionNet(2048,B = B,S = S,C = C)  # 4 conv, 2 linear
        )

    def forward(self, x):
        out = self.stem(x)
        return self.model.forward(out)


class DetectionNet(nn.Module):
    """The layers added on for detection as described in the paper."""

    def __init__(self, in_channels,B = 2,S = 13,C = 20):
        super().__init__()

        inner_channels = 512
        self.depth = (5 + C) * 5
        self.S = S
        self.B = B
        self.C = C
        self.model = nn.Sequential(
            nn.Conv2d(in_channels,inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1),  # (Ch, 14, 14) -> (Ch, 7, 7)
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels,out_channels=self.depth,kernel_size=1)
        )

    def forward(self, x):
        out = self.model(x)
        return out


def demo():
    model = YOLOv2ResNet(B = 2,S = 13,C = 20)
    x = torch.zeros(size=(1, 3, 416, 416))
    out = model(x)
    print(out.shape)

    # from torchinfo import summary
    # summary(model, input_size=(1, 3, 448, 448))


if __name__ == '__main__':
    demo()
    # backbone = resnet50(pretrained=True, progress=True)
    # model = nn.Sequential(
    #     backbone.conv1,
    #     backbone.bn1,
    #     backbone.relu,
    #     backbone.maxpool,
    #     backbone.layer1,
    #     backbone.layer2,
    #     backbone.layer3,
    #     backbone.layer4
    # )
    # x = torch.zeros(size = (1,3,416,416))
    # checkpoint = OrderedDict()
    # for name,param in list(backbone.named_parameters()):
    #     if name not in ['fc.weight','fc.bias']:
    #          checkpoint[name] = param
    # print('out.shape: {}'.format(model(x).shape))
    pass

