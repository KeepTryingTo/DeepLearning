"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/28 15:08
"""

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import cv2
from pathlib import Path
import torch
import torch.utils.data
import torchvision
import numpy as np
import dataset.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
            self, img_folder, ann_file,
            COCO_LABELS_id_2_name_MAP,COCO_LABELS_name_2_id_MAP,
            transforms,S = 7,B = 2,
            num_classes = 80,img_size = 448,
    ):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks=False)
        self.img_size = img_size
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.COCO_LABELS_id_2_name_MAP = COCO_LABELS_id_2_name_MAP
        self.COCO_LABELS_name_2_id_MAP = COCO_LABELS_name_2_id_MAP

    def x1y1x2y2Tocxcywh(self, boxes):
        x1, x2, y1, y2 = boxes
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return (cx, cy, w, h)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        img = img.resize((self.img_size,self.img_size))
        image = img
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        gt_map = torch.zeros(size=(self.S, self.S, 5 * self.B + self.num_classes))
        data_height, data_width = target['size']

        boxes = []
        labels = []
        for i,obj in enumerate(target["boxes"]):
            # 将所有的gt box信息转换成相对值0-1之间
            xmin = float(obj[0]) / data_width
            xmax = float(obj[2]) / data_width
            ymin = float(obj[1]) / data_height
            ymax = float(obj[3]) / data_height

            boxes.append([xmin, xmax, ymin, ymax])
            cls_name = self.COCO_LABELS_id_2_name_MAP[target['labels'][i].item()]
            cls_id = self.COCO_LABELS_name_2_id_MAP[cls_name]
            labels.append(cls_id)
        cell_size = 1 / self.S
        for i, (label, box) in enumerate(zip(labels, boxes)):
            class_label = int(label)
            # box的值已经进行了归一化处理
            cx, cy, w, h = self.x1y1x2y2Tocxcywh(boxes=box)
            ###########################  test the cx,cy,w,h #############################
            # x1,y1,x2,y2 = int((cx - w / 2) * self.img_size),\
            #               int((cy - h / 2) * self.img_size),\
            #               int((cx + w / 2) * self.img_size),\
            #               int((cy + h / 2) * self.img_size)
            #
            # cv2.rectangle(img=img,pt1=(x1,y1),pt2=(x2,y2),color=(0,255,255),thickness=1)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
            #############################################################################
            # 计算网格中心点的左上角坐标以及相对于左上角的中心坐标
            i, j = int(cx / cell_size), int(cy / cell_size)
            # 将其(j,i)转换为[0,1]之间的左上角
            x, y = i * cell_size, j * cell_size
            x_cell, y_cell = (cx - x) / cell_size, (cy - y) / cell_size
            #############################################################################
            # cx = x_cell * cell_size + x
            # cy = y_cell * cell_size + y
            # x1,y1,x2,y2 = int((cx - w / 2) * self.img_size),\
            #               int((cy - h / 2) * self.img_size),\
            #               int((cx + w / 2) * self.img_size),\
            #               int((cy + h / 2) * self.img_size)
            # cv2.rectangle(img=img,pt1=(x1,y1),pt2=(x2,y2),color=(0,255,255),thickness=1)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
            #############################################################################
            # 对框的高宽进行缩放
            # width_cell,height_cell = [
            #     w * self.S,
            #     h * self.S
            # ]
            width_cell, height_cell = [
                w, h
            ]
            if gt_map[j, i, class_label] == 0:
                # 得到变换之后的坐标
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                # coordinate
                gt_map[j, i, self.num_classes:self.num_classes + 4] = box_coordinates
                gt_map[j, i, self.num_classes + 5:self.num_classes + 9] = box_coordinates
                # confidence
                gt_map[j, i, self.num_classes + 4] = 1
                gt_map[j, i, self.num_classes + 9] = 1
                # class label
                gt_map[j, i, class_label] = 1
        if self._transforms is not None:
            image = self._transforms(image)
        return image, gt_map



class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        #对于重叠的目标过滤掉
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        #得到boxes
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing 转换为tensor: [num_boxes,4]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # 坐标表示 ：[xmin, ymin, width, height] ----> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        #得到类别id以及转换为tensor
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        #将存在的xmin > xmax,ymin > ymax过滤掉(注意：在标注的过程中会出现此问题)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, coco_path,transforms,COCO_LABELS_name_2_id_MAP,COCO_LABELS_id_2_name_MAP):
    root = Path(coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(
        img_folder, ann_file, transforms=transforms,
        COCO_LABELS_name_2_id_MAP = COCO_LABELS_name_2_id_MAP,
        COCO_LABELS_id_2_name_MAP = COCO_LABELS_id_2_name_MAP
    )
    return dataset

if __name__ == '__main__':
    from configs.config import *
    dataset_train = build(
        image_set='val', coco_path=COCO_PATH, transforms=transform,
        COCO_LABELS_name_2_id_MAP=COCO_LABELS_name_2_id_MAP,
        COCO_LABELS_id_2_name_MAP=COCO_LABELS_id_2_name_MAP
    )
    print(dataset_train.__len__())
    print(dataset_train[0])