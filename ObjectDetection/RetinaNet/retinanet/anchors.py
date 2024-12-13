import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None,
                 sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        #TODO 根据当前的层获得对应的步长
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        #TODO 根据层获得对应的anchor box尺寸大小
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        #TODO anchor box高宽比率
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        #TODO anchor box缩放尺度
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):
        
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        #TODO 获得对应层输出特征图大小
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            #TODO 这里只是根据anchor box的比例以及尺度生成了anchor box
            anchors         = generate_anchors(base_size=self.sizes[idx],
                                               ratios=self.ratios,
                                               scales=self.scales)
            #TODO
            shifted_anchors = shift(shape=image_shapes[idx],
                                    stride=self.strides[idx],
                                    anchors=anchors)
            all_anchors     = np.append(all_anchors,
                                        shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))

def generate_anchors(base_size=16,
                     ratios=None,
                     scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    #TODO 根据anchor的高宽比率以及其缩放尺度得到他们之间组合数=9
    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # TODO scale base_size np.tile 是 NumPy 库中的一个函数，用于重复数组。
    #  它的作用是将给定的数组沿着指定的轴重复指定的次数，从而创建一个更大的数组。
    #TODO 在dim=0维度重复2次，在dim=1维度重复len(ratios)次
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # TODO 计算anchor box的面积 compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # TODO correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(a = ratios, repeats = len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(a = ratios, repeats = len(scales))

    # TODO transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx],
                                           ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    #TODO 根据特征图大小以及步长获得对应X和Y轴坐标序列
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    #TODO ravel将多维展开为一维
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    #TODO 根据生成的anchor和网格坐标偏移获得对应anchor在网格上情况
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

