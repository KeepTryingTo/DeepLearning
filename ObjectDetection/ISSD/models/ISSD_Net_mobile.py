import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

from models.modules import InceptionModule

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
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

class BasicSepConv(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups = in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ISSDNet(nn.Module):

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(ISSDNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        if size == 300:
            self.indicator = 1
        else:
            print("Error: Sorry only RFB300_mobile is supported!")
            return
        #TODO backbone
        self.base = nn.ModuleList(base)
        #TODO backbone后部分的额外网络结构
        self.extras = nn.ModuleList(extras)

        #TODO 坐标框和置信度的预测head
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
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # TODO apply vgg up to conv4_3 relu backbone
        for k in range(12):
            x = self.base[k](x)

        #TODO torch.Size([2, 512, 19, 19])
        # print('x.shape: {}'.format(x.size()))
        sources.append(x)
        # print('s.shape: {}'.format(s.size())) torch.Size([2, 512, 19, 19])

        for k in range(12, len(self.base)):
            x = self.base[k](x)
        # TODO torch.Size([2, 1024, 10, 10])
        # print('x.shape: {}'.format(x.size()))
        sources.append(x)

        # TODO apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if (k + 1) % 3 == 0 and k > 0:
                # print('x.shape: {}'.format(x.size()))
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            l_conv = l(x).permute(0, 2, 3, 1).contiguous()
            c_conv = c(x).permute(0, 2, 3, 1).contiguous()
            # print('l.shape: {}'.format(l_conv.size()))
            loc.append(l_conv)
            conf.append(c_conv)

        #TODO [b,h,w,c] => [b,hwc] => cat(dim=1)
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


def conv_bn(inp,oup,stride):
    return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
            nn.Conv2d(inp,inp, kernel_size=3, stride=stride, padding=1,groups = inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
    )

def MobileNet():
    layers = []
    layers += [conv_bn(3, 32, 2)]
    layers += [conv_dw(32, 64, 1)]
    layers += [conv_dw(64, 128, 2)]
    layers += [conv_dw(128, 128, 1)]
    layers += [conv_dw(128, 256, 2)]
    layers += [conv_dw(256, 256, 1)]
    layers += [conv_dw(256, 512, 2)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 1024, 2)]
    layers += [conv_dw(1024, 1024, 1)]

    return layers



def add_extras(size, i, batch_norm=False):
    layers = []

    # TODO 10 x 10 inception module
    layers += [BasicConv(1024, 512, kernel_size=3, stride=2, padding=1)]
    layers += [InceptionModule(in_channels=512)] * 2

    # TODO 5 x 5 inception module
    layers += [BasicConv(512, 512, kernel_size=3, stride=2, padding=1)]
    layers += [InceptionModule(in_channels=512)] * 2

    #TODO 默认size = 300
    if size ==300:
        layers += [BasicConv(512,256,kernel_size=3,stride=2, padding=1)]
        layers += [BasicConv(256,256,kernel_size=1,stride=1)]
        layers += [BasicConv(256, 256, kernel_size=1, stride=1)]
        layers += [BasicConv(256, 256, kernel_size=3, stride=2, padding=1)]
        layers += [BasicConv(256, 256, kernel_size=1, stride=1)]
        layers += [BasicConv(256, 256, kernel_size=1, stride=1)]
    else:
        print("Error: Sorry only RFB300_mobile is supported!")
        return
    return layers

def multibox(size, base, extra_layers, cfg, num_classes):
    #TODO 其中cfg表示mbox = {'300': [6, 6, 6, 6, 4, 4]}
    loc_layers = []
    conf_layers = []
    base_net= [-2,-1]
    #TODO 分别针对backbone最后两层的输出结果
    for k, v in enumerate(base_net):
        if k == 0:
            loc_layers += [nn.Conv2d(512,
                                 cfg[k] * 4, kernel_size=1, padding=0)]
            conf_layers +=[nn.Conv2d(512,
                                 cfg[k] * num_classes, kernel_size=1, padding=0)]
        else:
            loc_layers += [nn.Conv2d(1024,
                                 cfg[k] * 4, kernel_size=1, padding=0)]
            conf_layers += [nn.Conv2d(1024,
                        cfg[k] * num_classes, kernel_size=1, padding=0)]
    i = 2
    for k, v in enumerate(extra_layers):
        if (k + 1) % 3== 0 and k > 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                 * 4, kernel_size=1, padding=0)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                  * num_classes, kernel_size=1, padding=0)]
            i +=1
    return base, extra_layers, (loc_layers, conf_layers)

#TODO 针对最后输出特征图每一个定位点的anchor数量
mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
}


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only RFB300_mobile is supported!")
        return

    return ISSDNet(phase, size, *multibox(
                                size = size, base=MobileNet(),
                                extra_layers=add_extras(size, i = 1024,batch_norm=False),
                                cfg=mbox[str(size)], num_classes=num_classes),
                                num_classes=num_classes)

def demo():
    net = build_net("train")
    x = torch.zeros(size=(2,3,300,300))
    outs = net(x)
    for out in outs:
        print('out.shape: {}'.format(out.size()))


if __name__ == '__main__':
    demo()
    pass