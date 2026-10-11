"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/11/12-19:34
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def normalize(feat, scale=20.0):
    norm = torch.sum(feat * feat, 1, keepdim=True)
    feat = feat / torch.sqrt(norm)
    feat = feat * scale
    return feat


class Normalize(nn.Module):

    def __init__(self, in_channel, across_spaticl, channel_shared):
        super(Normalize, self).__init__()
        self.spatial = across_spaticl
        self.channel = channel_shared
        if channel_shared:
            self.scale = Parameter(torch.FloatTensor(1))
        else:
            self.scale = Parameter(torch.FloatTensor(1, in_channel, 1, 1))
        init.constant_(self.scale, 20.0)

    def forward(self, feat):
        norm = torch.sum(feat * feat, 1, keepdim=True)
        if self.spatial:
            norm = torch.mean(torch.mean(norm, -1, keepdim=True), -2, keepdim=True)
        feat = feat / torch.sqrt(norm)
        feat = self.scale * feat
        return feat


class DDB_B(nn.Module):

    def __init__(self, n, g):
        super(DDB_B, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n, g, (1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(g, eps=0.001, momentum=0.001),

            nn.Conv2d(g, g, (3, 3), stride=1, padding=1, groups=g, bias=False),  # TODO 这里注意 第二个卷积是采用的是深度可分离卷积
            nn.BatchNorm2d(g, eps=0.001, momentum=0.001),
            nn.ReLU(True)
        )
        init.xavier_uniform_(self.conv[0].weight)
        init.xavier_uniform_(self.conv[2].weight)

    def forward(self, feat):
        inter = self.conv(feat)
        feat = torch.cat([feat, inter], dim=1)
        return feat


class Dense(nn.Module):

    def __init__(self, layers, n, g):
        super(Dense, self).__init__()
        self.dense = nn.ModuleList()
        for idx in range(layers):
            self.dense.append(DDB_B(n=n + idx * g, g=g))

    def forward(self, feat):
        for layer in self.dense:
            feat = layer(feat)
        return feat


class Transition(nn.Module):

    def __init__(self, in_channel, out_channel, pooling):
        super(Transition, self).__init__()
        self.pooling = pooling
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.001),
            nn.ReLU(True)
        )
        init.xavier_uniform_(self.conv[0].weight)
        if self.pooling:
            self.pool = nn.MaxPool2d((3, 3), stride=2, padding=1)

    def forward(self, feat):
        feat = self.conv(feat)
        if self.pooling:
            pool = self.pool(feat)
            return feat, pool
        return feat


class DownSample(nn.Module):

    def __init__(self, in_channel=128, multi=True):
        super(DownSample, self).__init__()
        self.multi = multi
        self.path_1 = nn.Sequential(
            nn.MaxPool2d((3, 3), stride=2, padding=1),
            nn.Conv2d(in_channel, 64, (1, 1), stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.001),
            nn.ReLU(True)
        )
        init.xavier_uniform_(self.path_1[1].weight)
        if self.multi:
            self.path_2 = nn.Sequential(
                nn.Conv2d(in_channel, 64, (1, 1),
                          stride=1, padding=0, bias=False),
                nn.Conv2d(64, 64, (3, 3), stride=2,
                          padding=1, groups=64, bias=False),  # TODO 注意这里采用的是深度可分离卷积
                nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.001),
                nn.ReLU()
            )
            init.xavier_uniform_(self.path_2[0].weight)
            init.xavier_uniform_(self.path_2[1].weight)

    def forward(self, feat):
        if self.multi:
            feat = torch.cat([self.path_1(feat), self.path_2(feat)], 1)
        else:
            feat = self.path_1(feat)
        return feat


class UpSample(nn.Module):

    def __init__(self, size, in_channel=128):
        super(UpSample, self).__init__()
        #        TODO self.upsample = nn.UpsamplingBilinear2d(size=size)
        self.size = size
        self.conv = nn.Conv2d(in_channel, 128, (3, 3),
                              stride=1, padding=1, groups=in_channel, bias=False)
        init.xavier_uniform_(self.conv.weight)

    def forward(self, feat):
        #       TODO  feat = self.upsample(feat)
        feat = F.interpolate(feat, size=self.size, mode='bilinear')
        feat = self.conv(feat)
        return feat


class FPN(nn.Module):

    def __init__(self):
        super(FPN, self).__init__()
        self.down_1 = DownSample(multi=False)
        self.down_2 = DownSample()
        self.down_3 = DownSample()
        self.down_4 = DownSample()
        self.down_5 = DownSample()
        self.up_1 = UpSample(3)
        self.up_2 = UpSample(5)
        self.up_3 = UpSample(10)
        self.up_4 = UpSample(19)
        self.up_5 = UpSample(38)
        self.conv = nn.Conv2d(128, 128,
                              (1, 1), stride=1, padding=0, bias=False)
        self.transform = nn.ReLU(True)
        init.xavier_uniform_(self.conv.weight)

    def forward(self, feat_1, feat_2):
        feat = [[]] * 11
        feat[0] = feat_1
        feat[1] = torch.cat([feat_2, self.down_1(feat_1)], 1)
        feat[2] = self.down_2(feat[1])
        feat[3] = self.down_3(feat[2])
        feat[4] = self.down_4(feat[3])
        feat[5] = self.down_5(feat[4])
        feat[6] = self.transform(self.up_1(feat[5]) + feat[4])
        feat[7] = self.transform(self.up_2(feat[6]) + feat[3])
        feat[8] = self.transform(self.up_3(feat[7]) + feat[2])
        feat[9] = self.transform(self.up_4(feat[8]) + feat[1])
        #        feat[10] = self.transform(self.up_5(feat[9]) + feat[0])
        feat[10] = self.transform(self.conv(self.up_5(feat[9])) + feat[0])
        return feat


class Predictor(nn.Module):

    def __init__(self, num_class):
        super(Predictor, self).__init__()
        self.num_class = num_class
        self.prior_boxes = [8, 8, 8, 8, 8, 8]
        self.normalize = nn.ModuleList()
        #        self.normalize = Normalize(128, False, False)
        self.locPredict = nn.ModuleList()
        self.clsPredict = nn.ModuleList()
        for i in range(len(self.prior_boxes) - 1, -1, -1):
            # TODO 定位层
            locPredictor = nn.Sequential(
                nn.Conv2d(128, 4 * self.prior_boxes[i], (1, 1),
                          stride=1, padding=0, bias=False),
                nn.Conv2d(4 * self.prior_boxes[i], 4 * self.prior_boxes[i], (3, 3),
                          stride=1, padding=1, groups=4 * self.prior_boxes[i], bias=False),
                nn.BatchNorm2d(4 * self.prior_boxes[i], eps=0.001, momentum=0.001)
            )
            init.normal_(locPredictor[0].weight, mean=0, std=0.01)
            self.locPredict.append(locPredictor)
            # TODO 分类层
            clsPredictor = nn.Sequential(
                nn.Conv2d(128, self.num_class * self.prior_boxes[i], (1, 1),
                          stride=1, padding=0, bias=False),
                nn.Conv2d(self.num_class * self.prior_boxes[i], self.num_class * self.prior_boxes[i],
                          kernel_size=(3, 3), stride=1, padding=1,
                          groups=self.num_class * self.prior_boxes[i], bias=False),
                nn.BatchNorm2d(self.num_class * self.prior_boxes[i], eps=0.001, momentum=0.001)
            )
            init.normal_(clsPredictor[0].weight, mean=0, std=0.1)
            self.clsPredict.append(clsPredictor)
            self.normalize.append(Normalize(128,
                                            False,
                                            False))

    def forward(self, feat):
        assert len(feat) == len(self.prior_boxes)
        loc = [[]] * len(self.prior_boxes)
        conf = [[]] * len(self.prior_boxes)
        for i in range(len(feat)):
            feat[i] = self.normalize[i](feat[i])

            loc[i] = self.locPredict[i](feat[i])
            conf[i] = self.clsPredict[i](feat[i])
            """
                loc.shape: torch.Size([1, 24, 2, 2])
                loc.shape: torch.Size([1, 24, 3, 3])
                loc.shape: torch.Size([1, 24, 5, 5])
                loc.shape: torch.Size([1, 24, 10, 10])
                loc.shape: torch.Size([1, 24, 19, 19])
                loc.shape: torch.Size([1, 24, 38, 38])
            """
            # print('loc.shape: {}'.format(loc[i].size()))

            loc[i] = loc[i].reshape([loc[i].shape[0], -1, 4])
            conf[i] = conf[i].reshape([conf[i].shape[0], -1, self.num_class])
        loc.reverse()
        conf.reverse()
        loc = torch.cat(loc, dim=1)
        conf = torch.cat(conf, dim=1)
        return loc, conf

class Framework(nn.Module):

    def __init__(self, num_class):
        super(Framework, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.001),
            nn.ReLU(True),

            nn.Conv2d(64, 64, (1, 1), stride=1,
                      padding=0, bias=False),
            nn.Conv2d(64, 64, (3, 3), stride=1,
                      padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.001),
            nn.ReLU(True),

            nn.Conv2d(64, 128, (1, 1), stride=1,
                      padding=0, bias=False),
            nn.Conv2d(128, 128, (3, 3), stride=1,
                      padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.001),
            nn.ReLU(True),

            nn.MaxPool2d((3, 3), stride=2, padding=1)
        )
        init.xavier_uniform_(self.stem[0].weight)
        init.xavier_uniform_(self.stem[3].weight)
        init.xavier_uniform_(self.stem[4].weight)
        init.xavier_uniform_(self.stem[7].weight)
        init.xavier_uniform_(self.stem[8].weight)
        self.dense_0 = Dense(4, 128, 32)
        self.transition_0 = Transition(256, 128, True)
        self.dense_1 = Dense(6, 128, 48)
        self.transition_1 = Transition(416, 128, True)
        self.dense_2 = Dense(6, 128, 64)
        self.transition_2 = Transition(512, 256, False)
        self.dense_3 = Dense(6, 256, 80)
        self.transition_3 = Transition(736, 64, False)
        self.fpn = FPN()
        self.predictor = Predictor(num_class)

    def init_model(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if 'bias' in m.state_dict().keys():
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print('Initializing weights for [extras, resblock,multibox]...')
        self.fpn.apply(weights_init)
        self.stem.apply(weights_init)
        self.predictor.apply(weights_init)
        self.conf.apply(weights_init)
        self.transition_0.apply(weights_init)
        self.transition_1.apply(weights_init)
        self.transition_2.apply(weights_init)
        self.transition_3.apply(weights_init)

    def forward(self, feat):
        feat = self.stem(feat)
        feat = self.dense_0(feat)
        _, feat = self.transition_0(feat)
        feat = self.dense_1(feat)
        first, feat = self.transition_1(feat)
        feat = self.dense_2(feat)
        feat = self.transition_2(feat)
        feat = self.dense_3(feat)
        feat = self.transition_3(feat)
        feat = self.fpn(first, feat)
        loc, conf = self.predictor(feat[5:])
        return loc, conf


def demo():
    x = torch.zeros(size=(1, 3, 300, 300))
    model = Framework(num_class=21)
    outs = model(x)

    """
    out.shape: torch.Size([1, 11658, 4])
    out.shape: torch.Size([1, 11658, 21])
    
    Total params: 946,860
    Trainable params: 946,860
    Non-trainable params: 0
    Total mult-adds (G): 1.12
    """
    for out in outs:
        print('out.shape: {}'.format(out.size()))

    from torchinfo import summary
    summary(model, input_size=(1, 3, 300, 300))


if __name__ == '__main__':
    # GtAnchorMatch()
    demo()
    pass
