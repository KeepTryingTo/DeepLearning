from __future__ import division
import torch
from math import sqrt as sqrt
from math import floor as floor
from itertools import product as product


class PriorBoxLayer(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    def __init__(self,width,height,
                 stride = [4,8,16,32,64,128], #TODO 输出的特征图大小相对于原图的下采样
                 box = [16,32,64,128,256,512], #TODO 设计的anchor box的尺寸
                 scale=[1,1,1,1,1,1], #TODO
                 aspect_ratios=[[], [], [], [], [], []]):
        super(PriorBoxLayer, self).__init__()
        self.width = width
        self.height = height
        self.stride = stride #TODO 下采样步长
        self.box = box #TODO 设定的anchor尺度大小
        self.scales = scale #TODO 缩放尺度
        #TODO 指定的anchor高宽比，由于默认人脸为正方形，因此只生成一个正方形anchor尺寸即可
        self.aspect_ratios = aspect_ratios
    def forward(self,prior_idx,f_width,f_height):
        mean = []

        for i in range(f_height):
            for j in range(f_width):
                for scale in range(self.scales[prior_idx]):
                    box_scale = (2**(1/3)) ** scale
                    cx = (j + 0.5) * self.stride[prior_idx] / self.width
                    cy = (i + 0.5) * self.stride[prior_idx] / self.height
                    #TODO 生成anchor box以及对其进行缩放
                    side_x = self.box[prior_idx]*box_scale / self.width
                    side_y = self.box[prior_idx]*box_scale / self.height
                    mean += [cx,cy,side_x,side_y]

                    #TODO 默认不再进行这里
                    for ar in self.aspect_ratios[prior_idx]:
                        mean += [cx,cy,side_x/sqrt(ar),side_y*sqrt(ar)]
        output = torch.Tensor(mean).view(-1, 4)
        return output
