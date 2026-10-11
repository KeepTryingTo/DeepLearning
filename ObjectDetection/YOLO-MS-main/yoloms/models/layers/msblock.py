# Copyright (c) MCG-NKU. All rights reserved.
from typing import Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmyolo.registry import MODELS
from mmdet.utils import OptConfigType
from mmcv.cnn import ConvModule

from ..utils import autopad


class MSBlockLayer(nn.Module):
    """MSBlockLayer

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int, tuple[int]): The kernel size of this Module.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and config norm layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer. Defaults to None.
    """
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: Union[int, Sequence[int]],
                 conv_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None) -> None:
        super().__init__()
        self.in_conv = ConvModule(in_channel,
                                  out_channel,
                                  1,
                                  conv_cfg=conv_cfg,
                                  act_cfg=act_cfg,
                                  norm_cfg=norm_cfg)        
        self.mid_conv = ConvModule(out_channel,
                                   out_channel,
                                   kernel_size,
                                   padding=autopad(kernel_size),
                                   groups=out_channel,
                                   conv_cfg=conv_cfg,
                                   act_cfg=act_cfg,
                                   norm_cfg=norm_cfg)
        self.out_conv = ConvModule(out_channel,
                                   in_channel,
                                   1,
                                   conv_cfg=conv_cfg,
                                   act_cfg=act_cfg, 
                                   norm_cfg=norm_cfg)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        x = self.in_conv(x)
        x = self.mid_conv(x)
        x = self.out_conv(x)
        return x
        


class MSBlock(nn.Module):
    """MSBlock

    Args:
        in_channel (int): The input channels of this Module.
        out_channel (int): The output channels of this Module.
        kernel_sizes (list(int, tuple[int])): Sequential of kernel sizes in MS-Block.
            
        in_expand_ratio (float): Channel expand ratio for inputs of MS-Block. Defaults to 3.
        mid_expand_ratio (float): Channel expand ratio for each branch in MS-Block. Defaults to 2.
        layers_num (int): Number of layer in MS-Block. Defaults to 3.
        in_down_ratio (float): Channel down ratio for downsample conv layer in MS-Block. Defaults to 1.
        
        attention_cfg (:obj:`ConfigDict` or dict, optional): Config dict for attention in MS-Block. Defaults to None.
        
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and config norm layer. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer. Defaults to dict(type='SiLU', inplace=True).
    """
    def __init__(self, 
                 in_channel: int,
                 out_channel: int,
                 kernel_sizes: Sequence[Union[int, Sequence[int]]],
                 
                 in_expand_ratio: float = 3.,
                 mid_expand_ratio: float = 2.,
                 layers_num: int = 3,
                 in_down_ratio: float = 1.,
                 
                 attention_cfg: OptConfigType = None,
                 conv_cfg: OptConfigType = None, 
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 ) -> None:
        super().__init__()
                
        self.in_channel = int(in_channel*in_expand_ratio)//in_down_ratio
        #TODO 根据卷积核的数量划分为几个分支的通道数
        self.mid_channel = self.in_channel//len(kernel_sizes)
        #TODO 中间通道数的膨胀率
        self.mid_expand_ratio = mid_expand_ratio
        groups = int(self.mid_channel*self.mid_expand_ratio)
        #TODO
        self.layers_num = layers_num
        self.in_attention = None
            
        self.attention = None
        if attention_cfg is not None:
            attention_cfg["dim"] = out_channel
            self.attention = MODELS.build(attention_cfg)

        
        self.in_conv = ConvModule(in_channels=in_channel,
                                  out_channels=self.in_channel,
                                  kernel_size=1,
                                  conv_cfg=conv_cfg,
                                  act_cfg=act_cfg,
                                  norm_cfg=norm_cfg)
        
        self.mid_convs = []
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [MSBlockLayer(in_channel=self.mid_channel,
                                    out_channel=groups,
                                    kernel_size=kernel_size,
                                    conv_cfg=conv_cfg,
                                    act_cfg=act_cfg,
                                    norm_cfg=norm_cfg) for _ in range(int(self.layers_num))]
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = ConvModule(in_channels=self.in_channel,
                                   out_channels=out_channel,
                                   kernel_size=1,
                                   conv_cfg=conv_cfg,
                                   act_cfg=act_cfg,
                                   norm_cfg=norm_cfg)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        # print('x.shape: {}'.format(x.size()))
        # print('in_channels: {}'.format(self.in_channel))
        out = self.in_conv(x)
        channels = []
        for i,mid_conv in enumerate(self.mid_convs):
            #TODO 根据划分的分支来进行不同分支的卷积操作
            channel = out[:,i*self.mid_channel:(i+1)*self.mid_channel,...]
            if i >= 1:
                #TODO 和上一个分支的卷积操作之后进行求和操作
                channel = channel + channels[i-1]
            channel = mid_conv(channel)
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        if self.attention is not None:
            out = self.attention(out)  
        return out

"""
    ### 1. `@MODELS.register_module()`
    这个装饰器的作用是将 `SE` 类注册到一个模块注册表中，使其可以在模型构建时使用。
    具体来说，这个注册表通常用于支持动态创建模型组件，而不需要直接在代码中实例化。
    
    通过这种方式，你可以以字符串形式将模型的名称传递给一个构建函数，例如在配置文件中，
    构建函数会根据注册名称（如 `"SE"`）创建相应的对象。
    
    ### 2. `from mmyolo.registry import MODELS` 的作用
    `mmyolo.registry` 是 `mmyolo` 库中的注册机制，它负责管理各种模型、损失函数和其他模块。
    通过导入 `MODELS`，你可以访问到这些注册模块进行动态构建。
    
    ### 3. `self.attention = MODELS.build(attention_cfg)`
    在这里，`self.attention` 是根据配置 `attention_cfg` 动态构建的一个模块。
    `attention_cfg` 包含了有关要构建模块的配置信息，其中可能指定了模型的类型（例如 `"SE"`）和初始化参数。
    
    `MODELS.build(attention_cfg)` 会：
    - 解析 `attention_cfg` 中的内容。
    - 查找 `attention_cfg` 中指定的模块名称（例如 'SE'）。
    - 利用之前注册的对应类（即你的 `SE`）创建模块的实例。
    - 将该实例赋值给 `self.attention`。
    
    ### 总结
    通过使用注册机制（如 `register_module`），你可以在代码中灵活构建模块和模型，
    而无需对每个模块进行硬编码。这提高了代码的可扩展性和可维护性，可以在不同的配置文件中灵活切换不同的模型模块。
    SE 注意力模块的这些动态注册和构建特点，使得该模块可以方便地集成到更大的模型中。

"""