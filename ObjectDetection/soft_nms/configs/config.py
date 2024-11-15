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

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = (320,320)
SHUFFLE = True
BATCH_SIZE = 8
VOC_NUM_CLASSES = 20
COCO_NUM_CLASSES = 80
NUM_WORKER = 8
BASE_LR = 0.0001
WEIGHT_DEACY = 1e-5
EPOCHS = 1000
STEP = 2
VAL_EPOCHS = 10
EPSILON = 1e-6
PIN_MEMORY = False
#由于本文是检测人脸，对于网络来说人这个类别很容易检测出来，
# 所以对应的置信度以及概率很高，因此设置的阈值也很高
CONF_THRESHOLD = 0.05
IOU_THRESHOLD = 0.1

#efficientdet
strides = [8, 16, 32, 64, 128]  # P3, P4, P5, P6 and P7
limit_ranges = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
head_only = False

#ssd configuration
anchors_size    = [30, 60, 111, 162, 213, 264, 315]
backbone = 'mobilenetv2'
#   Init_lr         模型的最大学习率;当使用Adam优化器时建议设置  Init_lr=6e-4;当使用SGD优化器时建议设置   Init_lr=2e-3
#   Min_lr          模型的最小学习率，默认为最大学习率的0.01
Init_lr = 2e-3
Min_lr = Init_lr * 0.01
optimizer_type = "sgd"
momentum = 0.937
weight_decay = 5e-4
lr_decay_type = 'cos'
save_period = 10
save_dir = 'weights'
#   eval_flag       是否在训练时进行评估，评估对象为验证集;安装pycocotools库后，评估体验更佳。
#   eval_period     代表多少个epoch评估一次，不建议频繁的评估;评估需要消耗较多的时间，频繁评估会导致训练非常慢
#   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
#   （一）此处获得的mAP为验证集的mAP。
#   （二）此处设置评估参数较为保守，目的是加快评估速度。
eval_flag = True
eval_period = 10
num_workers = 0
VOC_2012_PATH = r'/home/ff/myProject/KGT/myProjects/myDataset/voc2012/VOC2012'
VOC_2007_PATH = r'/home/ff/myProject/KGT/myProjects/myDataset/voc2012/VOC2007'
# VOC_PATH = r'/home/ff/YOLO/myDataset/VOCdevkit'
# COCO_PATH = r'/home/ff/YOLO/myDataset/MSCOCO/coco'


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

COCO_LABELS = [
 'person',
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

COCO_LABELS_id_2_name_MAP = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
 90: 'toothbrush'}
COCO_LABELS_name_2_id_MAP = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle':3, 'airplane':4,'bus':5, 'train':6, 'truck':7,
 'boat':8, 'traffic light':9, 'fire hydrant':10, 'stop sign':11, 'parking meter':12, 'bench':13,
 'bird':14, 'cat':15, 'dog':16, 'horse':17, 'sheep':18, 'cow':19, 'elephant':20, 'bear':21, 'zebra':22,
 'giraffe':23, 'backpack':24, 'umbrella':25, 'handbag':26, 'tie':27, 'suitcase':28, 'frisbee':29,
 'skis':30, 'snowboard':31, 'sports ball':32, 'kite':33, 'baseball bat':34, 'baseball glove':35,
 'skateboard':36, 'surfboard':37, 'tennis racket':38, 'bottle':39, 'wine glass':40, 'cup':41,
 'fork':42, 'knife':43, 'spoon':44, 'bowl':45, 'banana':46, 'apple':47, 'sandwich':48, 'orange':49,
 'broccoli':50, 'carrot':51, 'hot dog':52, 'pizza':53, 'donut':54, 'cake':55, 'chair':56, 'couch':57,
 'potted plant':58, 'bed':59, 'dining table':60, 'toilet':61, 'tv':62, 'laptop':63, 'mouse':64,
 'remote':65, 'keyboard':66, 'cell phone':67, 'microwave':68, 'oven':69, 'toaster':70, 'sink':71,
 'refrigerator':72, 'book':73, 'clock':74, 'vase':75, 'scissors':76, 'teddy bear':77, 'hair drier':78,
 'toothbrush':79}
