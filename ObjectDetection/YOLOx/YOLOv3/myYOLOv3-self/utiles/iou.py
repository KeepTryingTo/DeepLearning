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


def box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.
    box1, box2 are as: [xmin, ymin, xmax, ymax].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    '''
    N = box1.size(0)
    M = box2.size(0)

    #之所以转换为[N,M,2]其实就是相当于形成了一个[N,M]的表格进行计算
    lt = torch.max(
        box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = (rb-lt).clamp(min=0)  # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou