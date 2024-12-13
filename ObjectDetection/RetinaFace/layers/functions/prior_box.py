import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        #TODO 'min_sizes': [[16, 32], [64, 128], [256, 512]]
        self.min_sizes = cfg['min_sizes']
        #TODO 'steps': [8, 16, 32]
        self.steps = cfg['steps']
        #TODO 默认为false
        self.clip = cfg['clip']
        self.image_size = image_size
        #TODO 根据不同输出层的下采样步长获得对应特征图大小
        self.feature_maps = [[ceil(self.image_size[0]/step),
                              ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    #TODO 根据当前指定的anchor box大小归一化0-1之间
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    #TODO x * self.steps[k]首先将当前位置的anchor映射回原图坐标上，然后进行归一化处理
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    #TODO 根据中心以及anchor box尺寸得到anchor
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
