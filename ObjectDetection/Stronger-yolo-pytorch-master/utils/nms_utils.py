# coding: utf-8

from __future__ import division, print_function

import numpy as np
import torch
from utils.GIOU import iou_calc3


def torch_nms(cfg, boxes, variance=None):
    def nms_class(clsboxes):
        assert clsboxes.shape[1] == 5 or clsboxes.shape[1] == 9
        keep = []
        while clsboxes.shape[0] > 0:
            #TODO 获得当前预测类别分数最大分数
            maxidx = torch.argmax(clsboxes[:, 4])
            #TODO 对应box
            maxbox = clsboxes[maxidx].unsqueeze(0)
            #TODO 除了最大分数对应的box之外
            clsboxes = torch.cat((clsboxes[:maxidx], clsboxes[maxidx + 1:]), 0)
            #TODO 计算最大分数box和其他box之间的IOU
            iou = iou_calc3(maxbox[:, :4], clsboxes[:, :4])
            # TODO KL VOTE（variance vote）
            if variance is not None:
                ioumask = iou > 0
                #TODO 获得计算IOU大于0的box
                klbox = clsboxes[ioumask]
                klbox = torch.cat((klbox, maxbox), 0)
                #TODO 大于0的IOU值
                kliou = iou[ioumask]
                #TODO 获得预测的variance值
                klvar = klbox[:, -4:]
                pi = torch.exp(-1 * torch.pow((1 - kliou), 2) / cfg.vvsigma)
                pi = torch.cat((pi, torch.ones(1).cuda()), dim = 0).unsqueeze(1)
                pi = pi / klvar
                pi = pi / pi.sum(0)
                maxbox[0, :4] = (pi * klbox[:, :4]).sum(0)
            keep.append(maxbox)

            weight = torch.ones_like(iou)
            if not cfg.soft:
                weight[iou > cfg.nms_iou] = 0
            else:
                weight = torch.exp(-1.0 * (iou ** 2 / cfg.softsigma))
            clsboxes[:, 4] = clsboxes[:, 4] * weight
            filter_idx = (clsboxes[:, 4] >= cfg.score_thres).nonzero().squeeze(-1)
            clsboxes = clsboxes[filter_idx]
        return torch.cat(keep, 0).to(clsboxes.device)

    bbox = boxes[:, :4].view(-1, 4)
    numcls = boxes.shape[1] - 4
    scores = boxes[:, 4:].view(-1, numcls)
    # TODO 保存过滤之后的box，预测分数以及类别 Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []
    for i in range(numcls):
        #TODO 对当前类别低置信度的预测结果进行过滤
        filter_idx = (scores[:, i] >= cfg.score_thres).nonzero().squeeze(-1)
        if len(filter_idx) == 0:
            continue
        #TODO 获得过滤之后的box以及score
        filter_boxes = bbox[filter_idx]
        filter_scores = scores[:, i][filter_idx].unsqueeze(1)
        #TODO 如果采用varvote
        if variance is not None:
            filter_variance = variance[filter_idx]
            clsbox = nms_class(torch.cat((filter_boxes, filter_scores, filter_variance), 1))
        else:
            clsbox = nms_class(torch.cat((filter_boxes, filter_scores), 1))
        if clsbox.shape[0] > 0:
            picked_boxes.append(clsbox[:, :4])
            picked_score.append(clsbox[:, 4])
            picked_label.extend([torch.ByteTensor([i]) for _ in range(len(clsbox))])
    if len(picked_boxes) == 0:
        return None, None, None
    else:
        return torch.cat(picked_boxes), torch.cat(picked_score), torch.cat(picked_label)
