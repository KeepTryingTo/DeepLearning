import os
import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from utiles.container import Container


class VOCDataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, data_dir, split, transform=None,
                 target_transform=None, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        #VOCdevkit\VOC2012
        self.data_dir = data_dir
        #train or val
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        image_sets_file = os.path.join(self.data_dir,
                                       "ImageSets", "Main",
                                       "%s.txt" % self.split)

        #TODO 得到训练或者验证的图像名称
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        #TODO 是否保持keep_difficult = True or False
        self.keep_difficult = keep_difficult

        #TODO 得到类别名:类别索引
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        #TODO 得到图像名称
        image_id = self.ids[index]
        #TODO 根据XML文件得到图像中物体的boxes，labels以及difficult
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        #TODO Image读取对应的图像
        image = self._read_image(image_id)
        #TODO 进行数据增强
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        #TODO 对labels进行处理
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        return image, targets, index

    def __add__(self, other):
        if isinstance(other, VOCDataset):
            newDataset = self.ids + other.ids
            return self.__getitem__(newDataset)
        else:
            raise ValueError("Unsupported addition with the given type.")

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

    #读取图像的XML文件
    def _get_annotation(self, image_id):
        #得到标注文件
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % image_id)
        #得到XML文件标注物体的信息
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def get_img_info(self, index):
        img_id = self.ids[index]
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % img_id)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

from torch.utils.data.dataloader import default_collate
class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])

        if self.is_train:
            list_targets = transposed_batch[1]
            targets = Container(
                {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
            )
        else:
            targets = None
        return images, targets, img_ids