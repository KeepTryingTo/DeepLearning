"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/10/27-20:43
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes,
                 kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Feature_Fused_concat(nn.Module):
    def __init__(self,top_channels,bottom_channels):
        super(Feature_Fused_concat, self).__init__()
        self.top = nn.Sequential(
            BasicConv(in_planes=top_channels,out_planes=top_channels,
                      kernel_size=3,relu=True,
                      stride=1,padding=1)
        )
        self.bottom = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            BasicConv(in_planes=bottom_channels, out_planes=top_channels,
                      kernel_size=3,relu=True,
                      stride=1, padding=1)
        )
        self.final = nn.Sequential(
            BasicConv(in_planes=2 * top_channels, out_planes=top_channels,
                      kernel_size=1,stride=1,padding=0,bn=False,relu=True)
        )

    def forward(self,top_feature,bottom_feature):
        # print('top feature.shape: {}'.format(top_feature.size()))
        # print('bottom feature.shape: {}'.format(bottom_feature.size()))
        top = self.top(top_feature)
        bottom = self.bottom(bottom_feature)
        out = torch.cat((top,bottom),dim=1)
        out = self.final(out)
        return out

class Feature_Fused_sum(nn.Module):
    def __init__(self,top_channels,bottom_channels):
        super(Feature_Fused_sum, self).__init__()
        self.top = nn.Sequential(
            BasicConv(in_planes=top_channels,out_planes=top_channels,
                      kernel_size=3,relu=False,
                      stride=1,padding=1)
        )
        self.bottom = nn.Sequential(
            BasicConv(in_planes=bottom_channels, out_planes=top_channels,
                      kernel_size=1,relu=False,
                      stride=1, padding=0)
        )
        self.final = nn.ReLU()

    def forward(self,top_feature,bottom_feature):
        # print('top feature.shape: {}'.format(top_feature.size()))
        # print('bottom feature.shape: {}'.format(bottom_feature.size()))
        top = self.top(top_feature)
        size = top_feature.size()[2:]
        bottom_feature = F.interpolate(bottom_feature,size=size,
                                       mode='bilinear',align_corners=False)
        bottom = self.bottom(bottom_feature)
        # print('top.shape: {}'.format(top.size()))
        # print('bottom.shape: {}'.format(bottom.size()))
        out = top + bottom
        out = self.final(out)
        return out

class FFSSDNet(nn.Module):
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

    def __init__(self, phase, size, base,
                 extras, head, num_classes,is_concat = True):
        super(FFSSDNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return
        # vgg network
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)

        if is_concat:
            self.feature_fused = Feature_Fused_concat(
                top_channels=512,bottom_channels=512
            )
        else:
            self.feature_fused = Feature_Fused_sum(
                top_channels=512,bottom_channels=512
            )

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)
        conv_4_3 = x
        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        conv_5_3 = x

        sources.append(self.feature_fused(conv_4_3,conv_5_3))
        sources.append(conv_5_3)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k%2 ==0:
                # print('x.shape: {}'.format(x.size()))
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            l_conv = l(x).permute(0, 2, 3, 1).contiguous()
            c_conv = c(x).permute(0, 2, 3, 1).contiguous()
            # print('l.shape: {}'.format(l_conv.size()))
            loc.append(l_conv)
            conf.append(c_conv)

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
    # TODO fc6,fc7
    # conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    # conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    # layers += [pool5, conv6,
    #            nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    layers += [
        pool5
    ]
    return layers

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


#TODO SSD Layer
def add_extras(size):
    # Extra layers added to VGG for feature scaling
    layers = []
    #TODO 10 x 10
    layers += [BasicConv(512, 512, kernel_size=3, stride=2,padding=1)]
    #TODO 5 x 5
    layers += [BasicConv(512, 512, kernel_size=3, stride=2,padding=1)]
    #TODO 3x3
    layers += [BasicConv(512, 512, kernel_size=3, stride=2,padding=1)]
    layers += [BasicConv(512, 512, kernel_size=1, stride=1)]

    if size == 512:
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=4,stride=1,padding=1)]
    elif size ==300:
        layers += [BasicConv(512,512,kernel_size=3,stride=1)]
        layers += [BasicConv(512,512,kernel_size=1,stride=1)]
        layers += [BasicConv(512, 512, kernel_size=1, stride=1)]
        layers += [BasicConv(512, 128, kernel_size=1, stride=1)]
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return
    return layers

extras = {
    '300': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256,'S',256],
}


def multibox(size, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [-2]
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
    i = 1
    indicator = 0
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    for k, v in enumerate(extra_layers):
        if k < indicator or k%2== 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                 * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                  * num_classes, kernel_size=3, padding=1)]
            i +=1
    return vgg, extra_layers, (loc_layers, conf_layers)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=300, num_classes=21, is_concat = True):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    return FFSSDNet(phase, size,
                    *multibox(
                        size=size, vgg=vgg(base[str(size)], i=3),
                        extra_layers=add_extras(size),
                        cfg=mbox[str(size)], num_classes=num_classes
                    ),
                    num_classes,
                    is_concat=is_concat
                )

def demo():
    net = build_net("train",is_concat=False)
    x = torch.zeros(size=(2,3,300,300))
    outs = net(x)
    for out in outs:
        print('out.shape: {}'.format(out.size()))


if __name__ == '__main__':
    demo()
    pass