"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/12 19:15
"""

import torch
from torch import nn

def intersection_over_union(pred_boxes,gt_boxes):
    """
    :param pred_boxes: [B,7,7,4],box = [cx,cy,w,h]
    :param gt_boxes: [B,7,7,4],box = [cx,cy,w,h]
    :return:
    """
    #[cx,cy,w,h] => [xmin,ymin,xmax,ymax]
    pred_xmin = pred_boxes[...,0:1] - pred_boxes[...,2:3] / 2
    pred_xmax = pred_boxes[...,0:1] + pred_boxes[...,2:3] / 2
    pred_ymin = pred_boxes[...,1:2] - pred_boxes[...,3:4] / 2
    pred_ymax = pred_boxes[...,1:2] + pred_boxes[...,3:4] / 2

    gt_xmin = gt_boxes[...,0:1] - gt_boxes[...,2:3] / 2
    gt_xmax = gt_boxes[...,0:1] + gt_boxes[...,2:3] / 2
    gt_ymin = gt_boxes[...,1:2] - gt_boxes[...,3:4] / 2
    gt_ymax = gt_boxes[...,1:2] + gt_boxes[...,3:4] / 2

    #计算box的面积
    area1 = abs((pred_xmax - pred_xmin) * (pred_ymax - pred_ymin))
    area2 = abs((gt_xmax - gt_xmin) * (gt_ymax - gt_ymin))

    #计算交集
    xmin = torch.max(pred_xmin,gt_xmin)
    ymin = torch.max(pred_ymin,gt_ymin)
    xmax = torch.min(pred_xmax,gt_xmax)
    ymax = torch.min(pred_ymax,gt_xmax)

    intersection = abs((xmax - xmin).clamp(0) * (ymax - ymin).clamp(0))

    return intersection / (area2 + area1 - intersection + 1e-6)


