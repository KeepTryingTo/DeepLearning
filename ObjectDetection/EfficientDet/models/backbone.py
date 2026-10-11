# Author: Zylo117
import numpy as np
import torch
from torch import nn
from models.efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet
from models.efficientdet.utils import Anchors

#网络模型默认使用EfficientDet0(compound_coef = 0)
#TODO https://github.com/lukemelas/EfficientNet-PyTorch/releases
class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0,
                 load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        #TODO 对应的D0-Dx7的序号
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        #TODO 对应于D0-D7x中的每一个BiFPN中输入的channels:   0    1   2    3    4    5    6    7    8
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        #TODO 对于D0-D7的BiFPN重复的个数
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        #TODO 对应D0-D7x的输入图像的大小D0-D7x
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        #TODO 对于D0-D7x，reg和cls的SeparableConvBlock层重复的次数
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        #TODO 对于D0-D7x的feature map的anchor的尺度
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        #TODO anchor的高宽比
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        #TODO 得到feature map的cell对应的anchors数量
        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[
                BiFPN(
                        num_channels=self.fpn_num_filters[self.compound_coef],
                        conv_channels=conv_channel_coef[compound_coef],
                        first_time=True if i == 0 else False,
                        attention=True if compound_coef < 6 else False,
                        use_p8=compound_coef > 7
                    )for i in range(self.fpn_cell_repeats[compound_coef])
            ]
        )

        self.num_classes = num_classes
        self.regressor = Regressor(
           in_channels=self.fpn_num_filters[self.compound_coef],
            num_anchors=num_anchors,
           num_layers=self.box_class_repeats[self.compound_coef],
           pyramid_levels=self.pyramid_levels[self.compound_coef]
        )
        self.classifier = Classifier(
             in_channels=self.fpn_num_filters[self.compound_coef],
             num_anchors=num_anchors,
             num_classes=num_classes,
             num_layers=self.box_class_repeats[self.compound_coef],
             pyramid_levels=self.pyramid_levels[self.compound_coef]
        )
        """
        **kwarge:
            anchors的缩放尺度：
                anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
            anchor的高宽比：
                anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
        """
        self.anchors = Anchors(
           anchor_scale=self.anchor_scale[compound_coef],
           pyramid_levels=(
                   torch.arange(
                       self.pyramid_levels[self.compound_coef]
                   ) + 3
           ).tolist(),
           **kwargs
        )

        self.backbone_net = EfficientNet(
            compound_coef=self.backbone_compound_coef[compound_coef],
            num_classes=num_classes,
            load_weights=load_weights
        )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        """
            p3.shape: torch.Size([1, 40, 64, 64])
            p4.shape: torch.Size([1, 112, 32, 32])
            p5.shape: torch.Size([1, 320, 16, 16])
        """
        # print('p3.shape: {}'.format(p3.shape))
        # print('p4.shape: {}'.format(p4.shape))
        # print('p5.shape: {}'.format(p5.shape))

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype)

        return features, regression, classification, anchors

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')


def demoEfficientDet():
    efficientDet0 = EfficientDetBackbone(
        num_classes=20,
        compound_coef=0
    )

    x = torch.zeros(size = (1,3,512,512))
    outputs = efficientDet0(x)
    """
        P0_out.shape: torch.Size([1, 64, 64, 64])
        P1_out.shape: torch.Size([1, 64, 32, 32])
        P2_out.shape: torch.Size([1, 64, 16, 16])
        P3_out.shape: torch.Size([1, 64, 8, 8])
        P4_out.shape: torch.Size([1, 64, 4, 4])
        
        regression.shape: torch.Size([49104, 4])
        classification.shape: torch.Size([49104, 20])
        
        anchor.shape: torch.Size([49104, 4])
    """
    for i,out in enumerate(outputs[0]):
        print('P{}_out.shape: {}'.format(i,out.shape))
    print('regression.shape: {}'.format(np.shape(outputs[1][0])))
    print('classification.shape: {}'.format(np.shape(outputs[2][0])))
    print('anchor.shape: {}'.format(np.shape(outputs[3][0])))

    #total params: 3,839,117
    # from torchinfo import summary
    # summary(efficientDet0,input_size=(1,3,512,512))


if __name__ == '__main__':
    demoEfficientDet()
    pass
