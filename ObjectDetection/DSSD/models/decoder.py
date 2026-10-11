
import torch
import torch.nn as nn
from models.deconv_module import DeconvolutionModule

class DSSDDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        #(512, 1024, 1024, 1024, 1024, 1024) # 40, 20, 10, 5, 3, 1
        channels_backbone = cfg.MODEL.BACKBONE.OUT_CHANNELS
        #[1024, 512, 512, 512, 512, 512] # 1, 3, 5, 10, 20, 40
        channels_decoder = cfg.MODEL.DECODER.OUT_CHANNELS
        #[3, 1, 2, 2, 2]
        deconv_kernel_size = cfg.MODEL.DECODER.DECONV_KERNEL_SIZE
        #"prod"  # ["sum", "prod"]
        elementwise_type = cfg.MODEL.DECODER.ELMW_TYPE
        #定义空的列表用于记录卷积层
        self.decode_layers = nn.ModuleList()
        #得到backbone的最后卷积层的通道数
        cin_deconv = channels_backbone[-1]
        #由于采用转置卷积操作，所以channels_backbone[::-1][1:]翻转之后再从第一个卷积数读起
        for level, (cin_conv, cout) in enumerate(
                zip(channels_backbone[::-1][1:],
                    channels_decoder[1:])
        ):
            self.decode_layers.append(
                DeconvolutionModule(
                    cin_conv=cin_conv, cin_deconv=cin_deconv, cout=cout,
                    deconv_kernel_size=deconv_kernel_size[level],
                    elementwise_type=elementwise_type
                )
            )
            cin_deconv = cout

        self.num_layers = len(self.decode_layers)

    """
    features: [x, x5, x6, x7, x8, x9]
        out.shape: torch.Size([2, 512, 40, 40])
        out.shape: torch.Size([2, 1024, 20, 20])
        out.shape: torch.Size([2, 1024, 10, 10])
        out.shape: torch.Size([2, 1024, 5, 5])
        out.shape: torch.Size([2, 1024, 3, 3])
        out.shape: torch.Size([2, 1024, 1, 1])
    """
    def forward(self, features):
        features = list(features)
        for level in range(self.num_layers):
            x_deconv = features[-1-level]
            x_conv = features[-2-level]
            features[-2-level] = self.decode_layers[level](x_deconv, x_conv)

        return features


    