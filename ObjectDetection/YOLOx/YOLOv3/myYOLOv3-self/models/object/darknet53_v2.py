"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/11/27-9:48
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
import torch.nn as nn


model_urls = {
    "darknet53": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet53.pth",
}


__all__ = ['darknet53']


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class ResBlock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch//2, 1),
                Conv_BN_LeakyReLU(ch//2, ch, 3, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


class DarkNet_53(nn.Module):
    """
    DarkNet-53.
    """
    def __init__(self):
        super(DarkNet_53, self).__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, padding=1),
            Conv_BN_LeakyReLU(32, 64, 3, padding=1, stride=2),
            ResBlock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, padding=1, stride=2),
            ResBlock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, padding=1, stride=2),
            ResBlock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, padding=1, stride=2),
            ResBlock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, padding=1, stride=2),
            ResBlock(1024, nblocks=4)
        )


    def forward(self, x, targets=None):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        output = {
            'layer1': c3,
            'layer2': c4,
            'layer3': c5
        }

        return output


def build_darknet53(pretrained=False):
    # model
    model = DarkNet_53()

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['darknet53']
        # checkpoint state dict
        checkpoint_state_dict = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.object.modules import Conv


class YOLOv3(nn.Module):
    def __init__(self,
                 device,
                 num_classes=20,
                 trainable=False,
                 per_layer_num_anchors=3):
        super(YOLOv3, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.topk = 3000
        self.stride = [8, 16, 32]
        self.per_layer_num_anchors = per_layer_num_anchors

        # backbone
        self.backbone = build_darknet53(pretrained=trainable)

        # s = 32
        self.conv_set_3 = nn.Sequential(
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1)
        )
        self.conv_1x1_3 = Conv(512, 256, k=1)
        self.extra_conv_3 = Conv(512, 1024, k=3, p=1)
        self.pred_3 = nn.Conv2d(1024, self.per_layer_num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

        # s = 16
        self.conv_set_2 = nn.Sequential(
            Conv(768, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1)
        )
        self.conv_1x1_2 = Conv(256, 128, k=1)
        self.extra_conv_2 = Conv(256, 512, k=3, p=1)
        self.pred_2 = nn.Conv2d(512, self.per_layer_num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

        # s = 8
        self.conv_set_1 = nn.Sequential(
            Conv(384, 128, k=1),
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1),
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1)
        )
        self.extra_conv_1 = Conv(128, 256, k=3, p=1)
        self.pred_1 = nn.Conv2d(256, self.per_layer_num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

        self.init_yolo()

    def init_yolo(self):
        # Init head
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # init obj&cls pred
        for pred in [self.pred_1, self.pred_2, self.pred_3]:
            nn.init.constant_(pred.bias[..., :self.per_layer_num_anchors], bias_value)
            nn.init.constant_(pred.bias[..., self.per_layer_num_anchors: (1 + self.num_classes) * self.per_layer_num_anchors], bias_value)

    def forward(self, x, target=None):
        # backbone
        B = x.size(0)
        # backbone
        feats = self.backbone(x)
        c3, c4, c5 = feats['layer1'], feats['layer2'], feats['layer3']

        # FPN
        p5 = self.conv_set_3(c5)
        p5_up = F.interpolate(self.conv_1x1_3(p5), scale_factor=2.0, mode='bilinear', align_corners=True)

        p4 = torch.cat([c4, p5_up], 1)
        p4 = self.conv_set_2(p4)
        p4_up = F.interpolate(self.conv_1x1_2(p4), scale_factor=2.0, mode='bilinear', align_corners=True)

        p3 = torch.cat([c3, p4_up], 1)
        p3 = self.conv_set_1(p3)

        # head
        # s = 32
        p5 = self.extra_conv_3(p5)
        pred_3 = self.pred_3(p5)
        B,C,H_3,W_3 = pred_3.size()
        pred_3 = pred_3.view(B,self.per_layer_num_anchors,-1,H_3,W_3).permute(0,1,3,4,2)

        # s = 16
        p4 = self.extra_conv_2(p4)
        pred_2 = self.pred_2(p4)
        B, C, H_2, W_2 = pred_2.size()
        pred_2 = pred_2.view(B, self.per_layer_num_anchors, -1, H_2, W_2).permute(0,1,3,4,2)

        # s = 8
        p3 = self.extra_conv_1(p3)
        pred_1 = self.pred_1(p3)
        B, C, H_1, W_1 = pred_1.size()
        pred_1 = pred_1.view(B, self.per_layer_num_anchors, -1, H_1, W_1).permute(0,1,3,4,2)

        return pred_3,pred_2,pred_1

def demo():
    x = torch.randn(size=(1,3,416,416))
    model = YOLOv3(device='cpu',trainable=True,per_layer_num_anchors=3)
    outs = model(x)
    """
        out.size: torch.Size([B, 3, 13, 13, 25])
        out.size: torch.Size([B, 3, 26, 26, 25])
        out.size: torch.Size([B, 3, 52, 52, 25])
    """
    for out in outs:
        print('out.size: {}'.format(out.size()))

if __name__ == '__main__':
    demo()
    pass
