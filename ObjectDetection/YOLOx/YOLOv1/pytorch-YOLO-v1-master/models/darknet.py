"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/24 13:02
"""

import torch
import numpy as np
import torch.nn as nn
# from torchinfo import summary

"""
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding)
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(CNNBlock, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=False,
                **kwargs
            ),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self,x):
        out = self.conv(x)
        return out


class Yolov1(torch.nn.Module):
    def __init__(self,in_channels = 3,**kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layer(self.architecture)
        self.fcs = self._create_fcs(**kwargs)


    def _create_conv_layer(self,architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels,x[1],kernel_size = x[0],stride = x[2],padding = x[3]
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [torch.nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size = conv1[0],
                            stride = conv1[2],
                            padding = conv1[3]
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    ]
                    in_channels = conv2[1]

        return torch.nn.Sequential(*layers)

    def _create_fcs(self,split_size,num_boxes,num_classes):
        """
        :param split_size: 表示在原图上切分网格的大小
        :param num_boxes: 2
        :param num_classes:预测的类别数
        :return:
        """
        S,B,C = split_size,num_boxes,num_classes
        self.split_size, self.num_boxes, self.num_classes = (split_size,
                                                             num_boxes,
                                                             num_classes)
        """
        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))
        """
        fcs = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=1024 * 7 * 7,out_features=4096),
            torch.nn.Dropout(p = 0.0),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(in_features=4096,out_features=S * S * (C + B * 5))
        )
        return fcs

    def forward(self,x):
        b,c,h,w = x.size()
        x = self.darknet(x)
        """
        torch.nn.Flatten(start_dim=1,end_dim=-1)
        start_dim与end_dim代表合并的维度，开始的默认值为1，结束的默认值为-1(默认保留第0个维度)
        torch.flatten(t, start_dim=0, end_dim=-1)
        t表示的时要展平的tensor，start_dim是开始展平的维度，end_dim是结束展平的维度
        """
        out = self.fcs(torch.flatten(x,start_dim=1))
        out = out.view(b,self.split_size,self.split_size,(self.num_classes + self.num_boxes * 5))
        return out


def demoYolov1():
    model = Yolov1(in_channels=3,split_size = 14,num_boxes = 2,num_classes = 20)
    x = torch.randn(size = (1,3,448,448))
    out = model(x)
    #out.shape: [N,1470] = [N,S * S * (C + B * 5)]
    print('out.shape: {}'.format(out.shape))
    # summary(model,input_size=(1,3,448,448))

if __name__ == '__main__':
    # x = [[1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8]]
    # x = np.array(x)
    # """
    # PyTorch中的repeat()函数可以对张量进行重复扩充。
    # 当参数只有两个时：（列的重复倍数，行的重复倍数）。1表示不重复
    # 当参数有三个时：（通道数的重复倍数，列的重复倍数，行的重复倍数）。
    # """
    # y = torch.tensor(x).view(-1,2,4).repeat(1,2,2)
    # print('x.shape: {}'.format(np.shape(x)))
    # print('x.view: {}'.format(y.shape))
    # x = torch.arange(7).repeat(1,7,1).unsqueeze(dim=-1)
    # print(x.shape)
    demoYolov1()
    pass