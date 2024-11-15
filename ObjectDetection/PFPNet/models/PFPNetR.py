import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from data import data_configs

#TODO 定义的基准特征通道数大小
CHNS = 256


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 padding=0, bn_momentum=1e-3, affine=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class BottleneckConv(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=1e-3, affine=False):
        super(BottleneckConv, self).__init__()
        inter_channels = out_channels // 2
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, padding=0, bias=False),

            nn.BatchNorm2d(inter_channels, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class SerialConv(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=1e-3, drop_rate=0.1, affine=False):
        super(SerialConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),

            nn.BatchNorm2d(out_channels, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),

            nn.BatchNorm2d(out_channels, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace=True)
            #nn.Dropout2d(p=drop_rate)
        )

    def forward(self, x):
        return self.conv(x)


class PyramidHead(nn.Module):
    def __init__(self, cfg, max_size=40):
        super(PyramidHead, self).__init__()
        self.max_size = max_size
        # Pyramid Head Module
        layers = []
        for c in cfg:
            layers += [BasicConv(c, CHNS, kernel_size=1)]
        layers += [BasicConv(CHNS * len(cfg), CHNS * len(cfg))]
        self.convs = nn.ModuleList(layers)
        self.upsample = nn.Upsample(size=(max_size, max_size),
                                    mode='bilinear',
                                    align_corners=False)

    def forward(self, x_list):
        heads = list()
        for x, g in zip(x_list, self.convs[:-1]):
            x = g(x)
            #TODO 如果特征图大小小于指定的最大尺寸就上采样到指定大小
            if x.size(-1) < self.max_size:
                heads.append(self.upsample(x))
            else:
                heads.append(x)
        heads = torch.cat(heads, dim=1)
        return self.convs[-1](heads)


class PyramidPool(nn.Module):
    def __init__(self, head_channels, out_channels,
                 num_scales=4, affine=False):
        super(PyramidPool, self).__init__()
        self.num_scales = num_scales
        layers = []
        bns = []
        for _ in range(num_scales):
            layers += [BottleneckConv(head_channels, out_channels)]
            bns += [nn.BatchNorm2d(out_channels, momentum=1e-3, affine=affine)]
        self.convs = nn.ModuleList(layers)
        self.bns = nn.ModuleList(bns)

    def forward(self, x_list):
        pool = list()
        activated_pool = list()
        for x, g in zip(x_list, self.convs):
            pool.append(g(x))
            activated_pool.append(F.relu(pool[-1]))
        return pool, activated_pool


class MSCA(nn.Module):
    # refactoring
    def __init__(self, cfg, idx):
        super(MSCA, self).__init__()
        #TODO 得到当前要将特征图统一上采样或者下采样的指定大小
        self.size = cfg[idx]
        downsamples = []
        for n, _ in enumerate(cfg[:idx]):
            downsamples += [nn.AdaptiveAvgPool2d(output_size=self.size)]
        self.downsamples = nn.ModuleList(downsamples)
        upsamples = []
        for n, _ in enumerate(cfg[idx + 1:]):
            upsamples += [nn.Upsample(size=(self.size, self.size),
                                      mode='bilinear', align_corners=False)]
        self.upsamples = nn.ModuleList(upsamples)
        self.idx = idx
        self.convs = SerialConv(in_channels=CHNS * (len(cfg) - 1) * 2,
                                out_channels=CHNS)

    def forward(self, hd_list, ld_list):
        ftrs = list()
        #TODO 大的特征图下采样
        for l, d in zip(ld_list[:self.idx], self.downsamples):
            ftrs.append(d(l))
        ftrs.append(hd_list[self.idx])
        #TODO 小的特征图上采样
        for l, u in zip(ld_list[self.idx + 1:], self.upsamples):
            ftrs.append(u(l))
        #TODO 将采样到统一大小的特征图拼接
        ftrs = torch.cat(ftrs, dim=1)

        return self.convs(ftrs)


class SPP(nn.Module):
    def __init__(self, cfg):
        super(SPP, self).__init__()
        pools = []
        for s in cfg[1:]:
            pools.append(nn.AdaptiveAvgPool2d(output_size=s))
        self.pools = nn.ModuleList(pools)

    def forward(self, x):
        ftrs = [x]
        for p in self.pools:
            ftrs.append(p(x))

        return ftrs
# ---------------------------------------------------#
#                                                   #
#          Parallel Feature Pyramid Network         #
#                                                   #
# ---------------------------------------------------#

class PFPNet(nn.Module):
    # Define PFPNet with Modules
    def __init__(self, phase, size, num_classes,
                 backbone_, pyramid_head_, arm_, odm_,device = 'cpu'):
        super(PFPNet, self).__init__()
        # Parameters
        self.phase = phase
        self.device = device
        self.num_classes = num_classes
        self.size = size
        db = ('COCO', 'VOC')[num_classes == 21]
        self.cfg = data_configs[db][str(size)]
        self.num_pyramids = len(self.cfg['feature_maps'])
        self.priorbox = PriorBox(self.cfg,device)
        with torch.no_grad():
            self.priors = self.priorbox.forward()

        # PFP network
        self.vgg = nn.ModuleList(backbone_)
        self.pyramid_head = PyramidHead(pyramid_head_)
        self.spp = SPP(cfg=self.cfg['feature_maps'])
        self.fppool = PyramidPool(head_channels=CHNS * (self.num_pyramids - 1),
                                  out_channels=CHNS)
        #TODO 根据指定的融合特征图大小进行多尺度的融合
        self.msca = nn.ModuleList(
            MSCA(cfg=self.cfg['feature_maps'], idx=n)
            for n in range(len(self.cfg['feature_maps']))
        )

        # TODO Multibox Detector and Classifier (RefineDet version-anchor refinement module)
        self.arm_loc = nn.ModuleList(arm_[0])
        self.arm_conf = nn.ModuleList(arm_[1])
        self.odm_loc = nn.ModuleList(odm_[0])
        self.odm_conf = nn.ModuleList(odm_[1])

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect_RefineDet(num_classes, self.size,
                                       bkg_label=0, top_k=1000,
                                       conf_thresh=0.005,
                                       nms_thresh=0.005,
                                       objectness_thre=0.01,
                                       keep_top_k=500)

    def forward(self, x):
        # PFP feature extractors
        head_sources = list()
        pool_sources = list()
        odm_inputs = list()

        # Detector
        arm_loc = list()
        arm_conf = list()
        odm_loc = list()
        odm_conf = list()

        # ---------------------------------------------------#
        #             Backbone Network: VGG 16              #
        # ---------------------------------------------------#
        # apply vgg up to conv4_3: not rectified blob !!
        for k in range(22):  # 0~21
            x = self.vgg[k](x)

        head_sources.append(x)  # not rectified conv4_3 blob

        # apply vgg up to conv5_3: not rectified blob !!
        for k in range(22, 29):  # 0~28
            x = self.vgg[k](x)
        head_sources.append(x)  # not rectified conv5_3 blob
        # print('x.shape: {}'.format(x.size()))

        for k in range(29, len(self.vgg)):
            x = self.vgg[k](x)
        head_sources.append(x)
        # print('x.shape: {}'.format(x.size()))

        # TODO pyramid head（对于小于指定特征图大小的进行上采样到指定大小）并进行融合拼接
        x = self.pyramid_head(head_sources)  # pyramid head

        # TODO SPP 通过全局平均池化将特征图缩放到指定大小
        hd_sources = self.spp(x)

        # TODO FP 经过bottleneck层提取不同尺度的特征
        ld_sources, arm_inputs = self.fppool(hd_sources)

        # TODO MSCA 特征金字塔池化操作
        for m in self.msca:
            odm_inputs.append(m(hd_sources, ld_sources))

        # TODO apply ARM and ODM to source layers 锚点细化模块（ARM）预测的锚框（主要是anchor box进行一定的调整，
        #  调整之后的中心定位和高宽才能作为最终预测框的解码输出）
        for (x, l, c) in zip(arm_inputs, self.arm_loc, self.arm_conf):
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)

        # TODO apply ODM to source layers
        for (x, l, c) in zip(odm_inputs, self.odm_loc, self.odm_conf):
            odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)

        if self.phase == "test":
            output = self.detect(
                arm_loc.view(arm_loc.size(0), -1, 4),  # arm loc preds
                self.softmax(arm_conf.view(arm_conf.size(0), -1,2)),  # arm conf preds

                odm_loc.view(odm_loc.size(0), -1, 4),  # odm loc preds
                self.softmax(odm_conf.view(odm_conf.size(0), -1,
                                           self.num_classes)),  # odm conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, 2),
                odm_loc.view(odm_loc.size(0), -1, 4),
                odm_conf.view(odm_conf.size(0), -1, self.num_classes),
                self.priors
            )

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# Backbone Network
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        # add for pyramid head -> if 'v = D', inplace=False
        elif v == 'D':
            conv2d_diff = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d_diff, nn.BatchNorm2d(in_channels), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d_diff, nn.ReLU(inplace=False)]

        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7]

    return layers


# Multibox layers
def multibox(cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for k in range(4):
        loc_layers += [nn.Conv2d(in_channels=CHNS, out_channels=cfg[k] * 4,
                                 kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels=CHNS, out_channels=cfg[k] * num_classes,
                                  kernel_size=3, padding=1)]

    return loc_layers, conf_layers


# ---------------------------------------------------#
#               Configuration of Arch               #
# ---------------------------------------------------#
base = {
    '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 'D', 'M', 512, 512, 'D'],
    '512': [],
}

pyramid_head = [512, 512, 1024]

fp = {
    '320': ['pool40', 'pool20', 'pool10', 'pool5'],
    '512': [],
}

mbox = {
    '320': [3, 3, 3, 3],  # number of boxes per feature map location
    '512': [],
}


def build_pfp(phase, size=320, num_classes=21,device = 'cpu'):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 320:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only PFPNet320 (size=320) is supported!")
        return

    # Components for PFPNet
    backbone_ = vgg(base[str(size)], i=3)
    arm_ = multibox(mbox[str(size)], num_classes=2)
    odm_ = multibox(mbox[str(size)], num_classes=num_classes)

    return PFPNet(phase=phase, size=size,
                  num_classes=num_classes,
                  backbone_=backbone_,
                  pyramid_head_=pyramid_head,
                  arm_=arm_, odm_=odm_,device=device)
