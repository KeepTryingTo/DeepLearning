"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/9/29 13:18
"""
import os
import cv2
import torch
from torch.utils import data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

VOC_CLASS_LIST = ['__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']

class VOCDataset(data.Dataset):

    def __init__(self, root_dir, transform=None, target_transform=None,
                 keep_difficult=False, is_training=True):
        '''
        Dataset for PASCAL VOC
        :param root_dir: the root of the VOCdevkit
        :param split: which split (VOC2007 or VOC2012)
        :param transform: Image transforms
        :param target_transform: Box transforms
        :param keep_difficult: Keep difficult or not
        :param is_training: True if Training else False
        '''
        super(VOCDataset, self).__init__()
        if is_training:
            search_paths = os.path.join(root_dir,"ImageSets", "Main", "trainval.txt")
        else:
            search_paths = os.path.join(root_dir,"ImageSets", "Main", "test.txt")

        self.ids = VOCDataset._read_image_ids(search_paths)

        self.data_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult
        self.CLASSES = VOC_CLASS_LIST

        self.class_dict = {class_name: i for i, class_name in enumerate(self.CLASSES)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        annots, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            annots = annots[is_difficult == 0]
        image,img_height,img_width = self._read_image(image_id)

        samples = {'img':image,'annot':annots}
        if self.transform:
            samples = self.transform(samples)
        # return image, boxes, labels
        return samples

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % image_id)
        objects = ET.parse(annotation_file).findall("object")
        sample = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            label = self.class_dict[class_name] - 1

            sample.append([x1, y1, x2, y2,label])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)
        return (np.array(sample, dtype=np.float32),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        image = Image.open(image_file).convert("RGB")
        width,height = image.size
        image = np.array(image)
        return image,width,height


def demoVOC():
    vocDataset = VOCDataset(
        root_dir=r'E:\conda_3\PyCharm\Transer_Learning\PASCAL_VOC\VOCdevkit\VOC2012',
        transform=None,
        is_training=False
    )

    print('dataset.size: {}'.format(vocDataset.__len__()))
    print('dataset[0]: {}'.format(vocDataset[0]))

if __name__ == '__main__':
    demoVOC()
    pass


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
        # 计算image高宽比
        if height > width:
            # 计算scale = new_img / original_img
            scale = self.img_size / height
            resized_height = self.img_size
            # 对宽度进行同比例缩放
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width, :] = image

        # 对label进行同比例缩放
        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


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