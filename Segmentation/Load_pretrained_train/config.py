"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2026/10/11
@Desc   : 语义分割迁移学习训练框架——全局配置（调色板、模型注册表、常量）
"""

import numpy as np

# ImageNet 归一化参数（torchvision 分割模型骨干均为 ImageNet 预训练）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 分割标签中的忽略索引（VOC 边界白名单惯例）
IGNORE_INDEX = 255

# torchvision 预训练分割头的输出类别数（COCO-with-VOC-labels）
PRETRAIN_NUM_CLASSES = 21

# PASCAL VOC 21 类（第 1 行为背景，行号 = 类别 id）
VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
               "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]

# VOC 官方调色板（21 色）
VOC_PALETTE = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                        [0, 0, 128], [128, 0, 128], [0, 128, 128],
                        [128, 128, 128], [64, 0, 0], [192, 0, 0],
                        [64, 128, 0], [192, 128, 0], [64, 0, 128],
                        [192, 0, 128], [64, 128, 128], [192, 128, 128],
                        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                        [0, 64, 128]], dtype=np.uint8)

# 可加载的 torchvision 预训练分割模型（weights='DEFAULT'，均含 ImageNet 骨干预训练）
SEG_MODELS = {
    "fcn_resnet50": "FCN-ResNet50",
    "fcn_resnet101": "FCN-ResNet101",
    "deeplabv3_resnet50": "DeepLabV3-ResNet50",
    "deeplabv3_resnet101": "DeepLabV3-ResNet101",
    "deeplabv3_mobilenet_v3_large": "DeepLabV3-MobileNetV3-Large",
    "lraspp_mobilenet_v3_large": "LRASPP-MobileNetV3-Large",
}

# 损失函数可选项
LOSS_OPTIONS = ["ce", "dice", "focal", "ce_dice", "focal_dice"]

# 优化器可选项
OPTIMIZER_OPTIONS = ["sgd", "adam", "adamw"]

# 学习率调度可选项（poly 为分割标准策略）
SCHEDULER_OPTIONS = ["poly", "step", "cos", "none"]


def palette_for(num_classes):
    """任意类别数的调色板：VOC 21 色循环扩展"""
    if num_classes <= len(VOC_PALETTE):
        return VOC_PALETTE[:num_classes]
    return np.array([VOC_PALETTE[i % len(VOC_PALETTE)] for i in range(num_classes)],
                    dtype=np.uint8)
