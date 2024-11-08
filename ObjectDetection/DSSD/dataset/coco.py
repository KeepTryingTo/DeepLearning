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
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from utiles.encoder import Encoder
import dataset.myDir.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, anchors,transforms,S = 7,B = 2,num_classes = 80,img_size = 448):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks=False)
        self.img_size = img_size
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.anchors = anchors
        self.encoder = Encoder(
            anchors=self.anchors, img_size=self.img_size,
            S=self.S, B=self.B, num_classes=self.num_classes
        )

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        image_t = img.resize((self.img_size, self.img_size))
        h,w = target['size']


        if self._transforms is not None:
            img, target = self._transforms(img, target)
        boxes = []
        labels = []
        for i, obj in enumerate(target["boxes"]):
            # 将所有的gt box信息转换成相对值0-1之间
            xmin = float(obj[0]) / w
            xmax = float(obj[2]) / w
            ymin = float(obj[1]) / h
            ymax = float(obj[3]) / h

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: there are some bbox w/h <=0")
                continue
            boxes.append([xmin, xmax, ymin, ymax])
            labels.append(target['labels'][i])
        box_wh = torch.tensor([w, h, w, h], dtype=torch.float).expand_as(boxes)
        boxes /= box_wh
        loc_targets, cls_targets, boxes_target = self.encoder.encoder(boxes, labels)
        loc_targets, cls_targets, boxes_target = torch.as_tensor(loc_targets), \
                                                 torch.as_tensor(cls_targets), torch.as_tensor(boxes_target)
        return image_t, loc_targets, cls_targets, boxes_target

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


def build(image_set, coco_path):
    root = Path(coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set))
    return dataset
