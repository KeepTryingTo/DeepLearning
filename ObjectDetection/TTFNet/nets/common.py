from torch import nn
from mmcv.ops import DeformConv2dPack,ModulatedDeformConv2dPack


class CR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, padding=None, bias=True):
        super(CR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              stride, padding, bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x



class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, padding=None, bias=True):
        super(CBR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x



class CGR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CGR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.gn = nn.GroupNorm(32,out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        return x





#######################################################################################################


class DeconvLayer(nn.Module):

    def __init__(self, in_planes,out_planes, modulate_deform=True):
        super(DeconvLayer, self).__init__()
        if modulate_deform:
            self.dcn = ModulatedDeformConv2dPack(in_planes, out_planes,
                                                 kernel_size=3, padding=1,
                                                 deform_groups=1)
        else:
            self.dcn = DeformConv2dPack(in_planes, out_planes,
                                        kernel_size=3, padding=1,
                                        deform_groups=1)
        self.dcn_bn = nn.BatchNorm2d(out_planes)
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dcn(x)
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        return x


class ShortcutConv2d(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 conv_num):
        super(ShortcutConv2d, self).__init__()
        layers = []
        for i in range(conv_num):
            inc = in_channel if i == 0 else out_channel
            layers.append(nn.Conv2d(in_channels=inc,
                                    out_channels=out_channel,
                                    kernel_size=3, padding=1))
            if i < (conv_num-1):
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y


class CenternetDeconv(nn.Module):
    """
    利用deformable conv + transposed conv实现上采样
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self, channels, shortcut_in_channels, modulate_deform):
        super(CenternetDeconv, self).__init__()

        #TODO deformable convolution
        self.deconv1 = DeconvLayer(
            channels[0], channels[1],
            modulate_deform=modulate_deform,
        )
        self.deconv2 = DeconvLayer(
            channels[1], channels[2],
            modulate_deform=modulate_deform,
        )
        self.deconv3 = DeconvLayer(
            channels[2], channels[3],
            modulate_deform=modulate_deform,
        )
        self.shortcut1 = ShortcutConv2d(shortcut_in_channels[0],
                                        out_channel=channels[1], conv_num=1)
        self.shortcut2 = ShortcutConv2d(shortcut_in_channels[1],
                                        out_channel=channels[2], conv_num=2)
        self.shortcut3 = ShortcutConv2d(shortcut_in_channels[2],
                                        out_channel=channels[3], conv_num=3)

    def forward(self, xs):
        x2, x3, x4, x5 = xs
        x = self.deconv1(x5) + self.shortcut1(x4)   # 1/16
        x = self.deconv2(x) + self.shortcut2(x3)  # 1/8
        x = self.deconv3(x) + self.shortcut3(x2)   # 1/4
        return x










