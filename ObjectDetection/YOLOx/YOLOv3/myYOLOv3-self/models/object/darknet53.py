"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/6/27 10:42
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer

B-表示后面跟着重复次数的残差快
S-为尺度预测块，计算yolo损失
U-表示对特征图进行上采样，并与前一层进行连接
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    #-------------
    #CBL * 5 = 2 * CNNBlock + ResidualBlock + CNNBlock
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    #-------------
    #-------------
    #CBL + Upsample
    (256, 1, 1),
    "U",
    #-------------
    #CBL * 5 = 2 * CNNBlock + ResidualBlock + CNNBlock
    (256, 1, 1),
    (512, 3, 1),
    "S",
    #-------------
    #CBL + Upsample
    (128, 1, 1),
    "U",
    #-------------
    #CBL * 5 = 2 * CNNBlock + ResidualBlock + CNNBlock
    (128, 1, 1),
    (256, 3, 1),
    "S",
    #-------------
]

class CNNBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,bn_act = True,**kwargs):
        super(CNNBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels,bias=not bn_act,**kwargs)
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.leaky = torch.nn.LeakyReLU(negative_slope=0.2)
        self.use_bn_act = bn_act

    def forward(self,x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self,channels,use_residual=True,num_repeats = 1):
        super(ResidualBlock, self).__init__()
        self.layers = torch.nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += {
                torch.nn.Sequential(
                    CNNBlock(channels,channels // 2,kernel_size = 1),
                    CNNBlock(channels // 2,channels,kernel_size = 3,padding = 1)
                )
            }
        self.use_residual = use_residual
        self.num_repeats = num_repeats
    def forward(self,x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x

#CBL + Conv
class ScalePrediction(torch.nn.Module):
    def __init__(self,in_channels,num_classes):
        super(ScalePrediction, self).__init__()
        self.pred = torch.nn.Sequential(
            CNNBlock(in_channels,in_channels * 2,kernel_size = 3,padding = 1),
            CNNBlock(
                2 * in_channels,(num_classes + 5) * 3,bn_act=False,kernel_size = 1
            )
        )
        self.num_classes = num_classes

    def forward(self, x):
        out = self.pred(x)
        return out

class YOLOv3(torch.nn.Module):
    def __init__(self,in_channels = 3,num_classes = 80):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def _create_conv_layers(self):
        layers = torch.nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module,tuple):
                out_channels,kernel_size,stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride = stride,
                        padding = 1 if kernel_size == 3 else 0
                    )
                )
                in_channels = out_channels
            elif isinstance(module,list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(in_channels,num_repeats=num_repeats)
                )
            elif isinstance(module,str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels,use_residual=False,num_repeats = 1),
                        CNNBlock(in_channels,in_channels // 2,kernel_size = 1),
                        ScalePrediction(in_channels // 2,num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2
                elif module == 'U':
                    layers.append(torch.nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3
        return layers

    def forward(self,x):
        outputs = []
        route_connections = []
        for layer in self.layers:
            if isinstance(layer,ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)

            if isinstance(layer,ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer,torch.nn.Upsample):
                x = torch.cat([x,route_connections[-1]],dim = 1)
                route_connections.pop()
        return outputs

def demoYOLOv3():
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn(size = (2,3,IMAGE_SIZE,IMAGE_SIZE))
    out = model(x)
    print('yoloHead_13 x 13: {}'.format(out[0].shape))
    print('yoloHead_26 x 26: {}'.format(out[1].shape))
    print('yoloHead_52 x 52: {}'.format(out[2].shape))
    """
        out = [
            [BATCHSIZE, 3, 13, 13, 25]
            [BATCHSIZE, 3, 26, 26, 25]
            [BATCHSIZE, 3, 52, 52, 25]
        ]
    """
    from torchinfo import summary
    summary(model,input_size=(2,3,IMAGE_SIZE,IMAGE_SIZE))

if __name__ == '__main__':
    demoYOLOv3()
    pass