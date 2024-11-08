import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
from layers import *

import os
from models.modules import InceptionModule

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ISSDNet(nn.Module):
    """RFB Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1711.07767.pdf for more details on RFB Net.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(ISSDNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        # vgg network
        self.base = nn.ModuleList(base)
        # conv_4
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # TODO apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        #TODO torch.Size([2, 512, 38, 38])
        # print('x.shape: {}'.format(x.size()))
        sources.append(x)

        # TODO apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        #TODO torch.Size([2, 1024, 19, 19])
        # print('x.shape: {}'.format(x.size()))
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if (k + 1) % 3 ==0 and k > 0:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            location = l(x).permute(0, 2, 3, 1).contiguous()
            confidence = c(x).permute(0, 2, 3, 1).contiguous()
            loc.append(location)
            conf.append(confidence)
            # print('source.size: {}'.format(x.size()))
            # print('location.size: {}'.format(location.size()))

        #print([o.size() for o in loc])


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5,
               conv6,nn.ReLU(inplace=True),
               conv7, nn.ReLU(inplace=True)]
    return layers

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]
}


def add_extras(size, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    #TODO 10 x 10 inception module
    layers += [BasicConv(1024, 512, kernel_size=3, stride=2,padding=1)]
    layers += [InceptionModule(in_channels=512)] * 2

    # TODO 5 x 5 inception module
    layers += [BasicConv(512, 512, kernel_size=3, stride=2, padding=1)]
    layers += [InceptionModule(in_channels=512)] * 2

    #TODO 3 x 3 inception module
    layers += [BasicConv(512, 256, kernel_size=3, stride=2, padding=1)]
    layers += [InceptionModule(in_channels=256)] * 2

    if size == 300:
        layers += [BasicConv(256,256,kernel_size=3,stride=1)]
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=1,stride=1)]

    return layers

def multibox(size, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    #TODO conv5_3 fc7
    vgg_source = [-6,-2]
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers += [nn.Conv2d(512,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers +=[nn.Conv2d(512,
                                 cfg[k] * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    i = 2
    for k, v in enumerate(extra_layers):
        if (k + 1) % 3 == 0 and k > 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                 * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                  * num_classes, kernel_size=3, padding=1)]
            i +=1
    return vgg, extra_layers, (loc_layers, conf_layers)

mbox = {
    '300': [6, 6, 6, 6, 4, 4]  # number of boxes per feature map location
}

#TODO 这里实现的ISSD，暂时只支持size = 300
def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    return ISSDNet(phase, size, *multibox(size, vgg(base[str(size)], 3),
                                add_extras(size, i=1024, batch_norm=False),
                                mbox[str(size)], num_classes), num_classes)


def demo():
    net = build_net("train")
    x = torch.zeros(size=(2,3,300,300))
    outs = net(x)
    for out in outs:
        print('out.shape: {}'.format(out.size()))


if __name__ == '__main__':
    demo()
    pass