# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = '/home/ff/myProject/KGT/myProjects/myDataset/coco'
    DATASETS = {
        'coco_2014_train': {
            'img_dir': '/home/ff/myProject/KGT/myProjects/myDataset/coco/train2014/train2014',
            'ann_file': '/home/ff/myProject/KGT/myProjects/myDataset/coco/captions/annotations/instances_train2014.json'
        },
        'coco_2014_val': {
            'img_dir': '/home/ff/myProject/KGT/myProjects/myDataset/coco/val2014/val2014',
            'ann_file': '/home/ff/myProject/KGT/myProjects/myDataset/coco/captions/annotations/instances_val2014.json'
        },
        'coco_2017_test_dev': {
            'img_dir': '/home/ff/myProject/KGT/myProjects/myDataset/coco/val2014/val2014',
            'ann_file': '/home/ff/myProject/KGT/myProjects/myDataset/coco/captions/annotations/instances_val2014.json'
        },
        }

    @staticmethod
    def get(name):
        if 'coco' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs['img_dir']),
                ann_file=os.path.join(data_dir, attrs['ann_file']),
            )
            return dict(
                factory='COCODataset',
                args=args,
            )
        else:
            raise RuntimeError('Only support coco format now!')
        return None
