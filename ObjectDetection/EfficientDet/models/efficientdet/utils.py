import itertools
import torch
import torch.nn as nn
import numpy as np


class BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

        Returns:

        """
        #计算anchor box的center坐标
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]
        #将对predict_box的width和height进行转换
        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha
        #将对predict_box的center_x和center_y进行转换
        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a

        #[x,y,w,h] => [xmin,ymin,xmax,ymax]
        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape
        #对预测得到boxes进行边界处理
        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes


class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117
    """

    """
    pyramid_level = [3,4,5,6,7]
    **kwarge:
       anchors的缩放尺度：
           anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
       anchor的高宽比：
           anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
    """

    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        #TODO stride: [8,16,32,64,128],表示下采样的步长
        self.strides = kwargs.get(
            'strides', [2 ** x for x in self.pyramid_levels]
        )
        #todo scales: [1,1.3,1.6] 表示anchor box缩放的系数
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        #TODO [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])

        self.last_anchors = {}
        self.last_shape = None

    def forward(self, image, dtype=torch.float32):
        """Generates multiscale anchor boxes.

        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

        Returns:
          anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """
        #得到图像的高宽
        image_shape = image.shape[2:]

        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]

        #last_shape = image_shape
        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape

        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        boxes_all = []
        #TODO 对于每一个下采样倍数进行遍历
        for stride in self.strides:
            boxes_level = []
            #TODO 针对每一个anchor的尺度和anchor的高宽比进行遍历:
            # https://blog.csdn.net/The_Time_Runner/article/details/90143662
            for scale, ratio in itertools.product(self.scales, self.ratios):
                #TODO 保证图像大小为2的倍数
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                #TODO anchor_scale = 4,stride = 8,scale = 1 => 得到当前的anchor的大小
                base_anchor_size = self.anchor_scale * stride * scale
                #TODO 得到anchor的center_x
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                #TODO 得到anchor的center_y
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                #TODO x = [start = stride / 2,end = img_width,step = stride]，生成网格坐标
                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y) #xv,yv => [64,64]
                xv = xv.reshape(-1) #xv: [64 x 64]
                yv = yv.reshape(-1)

                # TODO y1,x1,y2,x2 ,生成一个将anchor覆盖在网格上的feature map; [4,4096]
                #TODO 将anchor box放置到网格上
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1) #[4096,4]
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))
        #[49104,4]
        anchor_boxes = np.vstack(boxes_all)

        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)

        # save it for later use to reduce overhead
        self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes
