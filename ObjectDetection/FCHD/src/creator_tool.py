# -*- coding: utf-8 -*- 

import numpy as np
import torch
# import cupy as cp
from torchvision.ops import nms
from src.bbox_tools import bbox2loc, bbox_iou, loc2bbox
# from src.nms import non_maximum_suppression
import src.array_tool as at

class AnchorTargetCreator(object):
    def __init__(self,
                 n_sample=32,
                 pos_iou_thresh=0.7,
                 neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        img_H, img_W = img_size
        n_anchor = len(anchor)
        #TODO 根据图像高宽进行anchor box边界的检测
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]

        #TODO 根据正样本以及负样本得到label
        argmax_ious, label = self._create_label(
            inside_index=inside_index,
            anchor=anchor,
            bbox=bbox
        )
        #TODO 根据anchor box对其gt box进行编码
        loc = bbox2loc(anchor, bbox[argmax_ious])
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)
        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)
        #TODO 计算gt box和anchor box之间的iou
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)

        #TODO 小于指定IOU阈值的作为背景
        label[max_ious < self.neg_iou_thresh] = 0
        #TODO 和gt box计算IOU最大的那些anchor box对应索引位置标签作为前景
        label[gt_argmax_ious] = 1
        #TODO 计算的IOU大于指定阈值的作为前景
        label[max_ious >= self.pos_iou_thresh] = 1
        #TODO 正样本数
        n_pos = int(self.pos_ratio * self.n_sample)
        #TODO 正样本索引
        pos_index = np.where(label == 1)[0]
        #TODO 如果正样本索引计算数量大于指定正样本数量，就随机的把len(pos_index) - n_pos置为背景
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1
        #TODO 计算负样本数量
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        #TODO 和正样本执行操作一样
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        ious = bbox_iou(anchor, bbox)
        #TODO 和anchor box计算IOU最大的那些gt box索引以及对应IOU
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        #TODO 和gt box计算IOU最大的那些anchor box以及对应IOU
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


class ProposalCreator:
    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=300,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self,
                 loc,
                 score,
                 anchor,
                 img_size,
                 scale=1.):

        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        #TODO 根据anchor box以及预测的loc对其进行解码得到[xmin,ymin,xmax,ymax]
        roi = loc2bbox(anchor, loc)

        #TODO 根据box随机的裁剪
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])
        
        #TODO 裁剪之后计算box的高宽
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        #TODO 边界处理


        keep = np.where((torch.tensor(hs) >= min_size) & (torch.tensor(ws) >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # keep = non_maximum_suppression(
        #     cp.ascontiguousarray(cp.asarray(roi)),
        #     thresh=self.nms_thresh)
        keep = nms(boxes=torch.tensor(roi),
                   scores=torch.tensor(score),
                   iou_threshold=self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep,:]
        score = score[keep]
        return roi, score
