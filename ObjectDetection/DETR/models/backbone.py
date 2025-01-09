# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.models import ResNet50_Weights
from torchvision.models import resnet50

from typing import Dict, List
from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding

# torchvision.models.resnet50

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        #rsqrt(): https://blog.csdn.net/weixin_37490132/article/details/126797344
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool,
                 num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        #TODO 由于没有训练分割模型，因此return_layers = {'layer4': "0"}，最后backbone只返回最后的结果，不返回中间结果
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        #backbone提取特征
        #TODO 其中tensor_list是经过打包之后的结果，包含: tensor,mask
        #TODO 其中tensor表示经过padding之后的图像已经改图像对应的mask（img有值的地方为false，否则为true）
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        #TODO 根据是否返回中间结果
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            #TODO 将mask缩放至和经过backbone之后的tensor
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            #TODO 经过backbone之后重新打包
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        #TODO 采用resnet50作为backbone，冻结BN层
        # backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation],
        #                                              weights=ResNet50_Weights.IMAGENET1K_V1, norm_layer=FrozenBatchNorm2d)
        backbone = getattr(torchvision.models,name)(pretrained = True,progress = True,
                                                    replace_stride_with_dilation=[False, False, dilation],
                                                    norm_layer=FrozenBatchNorm2d)
        #TODO 如果为resnet18或者resnet34的话，最后输出的通道数为512
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone,
                         num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        #TODO self[0]表示backbone输出的结果
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # self[1]： position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    #TODO 构建位置编码
    position_embedding = build_position_encoding(args)
    #TODO 定义的学习率如果大于0的话，那么对backbone进行训练
    train_backbone = args.lr_backbone > 0
    #TODO 对于分割的时候提供
    return_interm_layers = args.masks
    #TODO dilation如果使用的话，表示在最后一层卷积使用空洞卷积
    backbone = Backbone(args.backbone, train_backbone,
                        return_interm_layers, args.dilation)
    #TODO 结合位置编码
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
