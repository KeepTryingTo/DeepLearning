"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/11/16-10:41
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import torch
import numpy as np

import cv2

# 将一个batch中的所有图像以及标注信息填充到相同，以便于送入到模型中训练
def collater(data):
    imgs = [
        s['img'] for s in data
    ]
    annots = [
        s['annot'] for s in data
    ]  # scales: [0.8,0.8,0.8,0.8,...,0.8] len => batchSize
    scales = [
        s['scale'] for s in data
    ]
    # [batch,img_width,img_height,channels = 3]
    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    # 计算一个batch中所有图像中包含的最多Bbox数量
    max_num_annots = max(annot.shape[0] for annot in annots)
    if max_num_annots > 0:
        # annot_padded = [batch,max_num_annots,5] 所有的值都为-1
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
    # [batch,channels,img_width,img_height]
    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        # TODO 计算image高宽比
        if height > width:
            # 计算scale = new_img / original_img
            scale = self.img_size / height
            resized_height = self.img_size
            # TODO 对宽度进行同比例缩放
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height),
                           interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width, :] = image

        # 对gt box进行同比例缩放
        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32),
                'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            # image水平翻转
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            # label水平翻转
            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
