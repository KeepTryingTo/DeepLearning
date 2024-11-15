from itertools import product

import torch
from math import sqrt


class PriorBox:
    def __init__(self, cfg):
        #得到输入图像的大小
        self.image_size = cfg.INPUT.IMAGE_SIZE
        #得到prior anchors的配置项
        prior_config = cfg.MODEL.PRIORS
        #得到输出得到的特征图大小: [40, 20, 10, 5, 3, 1]
        self.feature_maps = prior_config.FEATURE_MAPS
        """
          MIN_SIZES是用于计算默认框大小的参数之一。根据引用中的解释，
          MIN_SIZES是根据公式Sk = Smin + (Smax - Smin) * (k - 1) / (m - 1)计算得出的。
          其中，Sk表示每个特征层的先验框大小与原图片大小之比，Smin和Smax分别表示最小和最大的比例。
          在SSD中，一共有6个用于分类和回归的特征层，即m=6。因此，根据公式，我们可以计算出MIN_SIZES
          的值为60, 111, 162, 213, 264, 315。这些值代表了不同特征层对应的默认框的尺寸。
        """
        """
            _C.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]
            _C.MODEL.PRIORS.MAX_SIZES = [60, 111, 162, 213, 264, 315]
        """
        self.min_sizes = prior_config.MIN_SIZES
        self.max_sizes = prior_config.MAX_SIZES
        #才采样步长: [8, 16, 32, 64, 107, 320]
        self.strides = prior_config.STRIDES
        #对应于每一个head的prior anchor的比率[[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.aspect_ratios = prior_config.ASPECT_RATIOS
        #是否进行裁剪
        self.clip = prior_config.CLIP

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        #遍历每一个特征图大小
        for k, f in enumerate(self.feature_maps):
            #计算得到对应输出大小的特征图
            scale = self.image_size / self.strides[k]
            #对得到特征图矩阵大小进行遍历
            for i, j in product(range(f), repeat=2):
                # unit center x,y 得到对应prior anchor的中心坐标
                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale

                # small sized square box
                size = self.min_sizes[k]
                #得到对应的min_prior_anchors在原图上的映射
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # big sized square box
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
