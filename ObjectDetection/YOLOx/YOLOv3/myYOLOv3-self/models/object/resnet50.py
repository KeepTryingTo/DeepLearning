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


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))

#################################
#       Transfer Learning       #
#################################
class YOLOv1ResNet(nn.Module):
    def __init__(self,B = 2,S = 7,C = 20):
        super().__init__()
        self.depth = B * 5 + C

        # Load backbone ResNet
        backbone = resnet50(pretrained=True,progress=True)
        backbone.requires_grad_(False)  # Freeze backbone weights

        # Delete last two layers and attach detection layers
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()

        self.model = nn.Sequential(
            backbone,
            Reshape(2048, 14, 14),
            DetectionNet(2048,B = B,S = S,C = C)  # 4 conv, 2 linear
        )

    def forward(self, x):
        return self.model.forward(x)


class DetectionNet(nn.Module):
    """The layers added on for detection as described in the paper."""

    def __init__(self, in_channels,B = 2,S = 7,C = 2):
        super().__init__()

        inner_channels = 512
        self.depth = 5 * B + C
        self.S = S
        self.B = B
        self.C = C
        self.model = nn.Sequential(
            nn.Conv2d(in_channels,inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=2, padding=1),  # (Ch, 14, 14) -> (Ch, 7, 7)
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Flatten(),

            nn.Linear(7 * 7 * inner_channels, 512),
            # nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(512, S * S * self.depth)
        )

    def forward(self, x):
        out = torch.reshape(
            self.model.forward(x),
            (-1, self.S, self.S, self.depth)
        )
        out = torch.sigmoid(out)
        return out


def demo():
    model = YOLOv1ResNet(B = 2,S = 7,C = 20)
    x = torch.zeros(size=(1, 3, 448, 448))
    out = model(x)
    print(out.shape)

    # from torchinfo import summary
    # summary(model, input_size=(1, 3, 448, 448))


if __name__ == '__main__':
    demo()
    pass

