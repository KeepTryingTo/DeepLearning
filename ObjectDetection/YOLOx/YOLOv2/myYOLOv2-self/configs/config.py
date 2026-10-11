"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/11 22:41
"""

import os
import numpy as np
import torch.cuda
from torchvision import transforms

DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 416
SHUFFLE = True
BATCH_SIZE = 8
S = 13
B = 5
NUM_CLASSES = 2
VOC_NUM_CLASSES = 20
NUM_WORKER = 0
BASE_LR = 0.0001
WEIGHT_DEACY = 1e-5
EPOCHS = 1000
STEP = 5
lambda_coord = 1.
lambda_noobj = .5
lambda_prior = 1.
lambda_obj = 1.
lambda_class = 1.
VAL_STEP = 2
EPSILON = 1e-6
PIN_MEMORY = False
#TODO anchors的高宽已经缩放至416大小的量纲
ANCHORS = (17,31, 44,65, 80,146, 170,236, 346,339)
#TODO 由于本文是检测人脸，对于网络来说人这个类别很容易检测出来，
# 所以对应的置信度以及概率很高，因此设置的阈值也很高
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.1
CHANNELS_LIST=(64, 192, 256, 512, 1024)
TRAIN_DIR_IMG = r'E:\Data(D)\workspace\max\OK\train\person\train_img'
TRAIN_DIR_LAB = r'E:\Data(D)\workspace\max\OK\train\person\train_txt'
VAL_DIR_IMG = r'E:\Data(D)\workspace\max\OK\train\person\test_img'
VAL_DIR_LAB = r'E:\Data(D)\workspace\max\OK\train\person\test_txt'

# VOC_PATH = r'E:\conda_3\PyCharm\Transer_Learning\PASCAL_VOC'
VOC_PATH = r'/home/ff/myProject/KGT/myProjects/myDataset/voc2012'
# COCO_PATH = r'E:\conda_3\PyCharm\Transer_Learning\MSCOCO\coco'
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

class ColorJitter(object):
    """对图像颜色信息进行随机调整,该方法应放在ToTensor前"""
    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05):
        self.trans = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, target):
        image = self.trans(image)
        return image, target

class Normalization(object):
    """对图像标准化处理,该方法应放在ToTensor后"""
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image, target):
        image = self.normalize(image)
        return image, target

transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])
PERSON_CLASS = {
    0:'background',
    1:'person'
}
VOC_CLASSES = {    # always index 0
    0:'aeroplane',
    1:'bicycle',
    2:'bird',
    3:'boat',
    4:'bottle',
    5:'bus',
    6:'car',
    7:'cat',
    8:'chair',
    9:'cow',
    10:'diningtable',
    11:'dog',
    12:'horse',
    13:'motorbike',
    14:'person',
    15:'pottedplant',
    16:'sheep',
    17:'sofa',
    18:'train',
    19:'tvmonitor'
}

COCO_LABELS = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]



if __name__ == '__main__':
    anchors = []
    for i in range(0,len(ANCHORS),2):
        # 由于给定的anchor未进行归一化处理;将其anchor的大小缩放至[0,S]之间的大小
        anchor_wh = (ANCHORS[i] / 416 * S, ANCHORS[i + 1] / 416 * S)
        anchors.append(anchor_wh)
    print(anchors)