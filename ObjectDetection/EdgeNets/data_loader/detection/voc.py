# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import os
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

    def __init__(self, root_dir, split, transform=None,
                 target_transform=None, keep_difficult=False,
                 is_training=True):
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
            search_paths = os.path.join(root_dir, split, "ImageSets", "Main", "trainval.txt")
        else:
            search_paths = os.path.join(root_dir, split, "ImageSets", "Main", "test.txt")

        self.ids = VOCDataset._read_image_ids(search_paths)

        self.data_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult
        self.CLASSES = VOC_CLASS_LIST

        self.class_dict = {class_name: i for i, class_name in enumerate(self.CLASSES)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

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
        annotation_file = os.path.join(self.data_dir,
                                       self.split, "Annotations", "%s.xml" % image_id)
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

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, self.split, "JPEGImages", "%s.jpg" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
