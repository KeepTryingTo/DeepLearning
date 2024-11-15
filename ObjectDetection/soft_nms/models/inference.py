"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/2/26 18:17
"""

import torch

from utiles.container import Container
from utiles.nms import boxes_nms
from utiles.nms_ import soft_nms


class PostProcessor:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.width = cfg.INPUT.IMAGE_SIZE
        self.height = cfg.INPUT.IMAGE_SIZE
        self.is_soft_nms = cfg.TEST.SOFT_NMS

    def __call__(self, detections):
        batches_scores, batches_boxes = detections
        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        #TODO 针对batch进行遍历
        for batch_id in range(batch_size):
            processed_boxes = []
            processed_scores = []
            processed_labels = []

            #TODO 得到对应batch的scores[N,cls]以及boxes[N,4]
            per_img_scores, per_img_boxes = batches_scores[batch_id], batches_boxes[batch_id]
            #TODO 跳过检测的背景并对所有的box进行遍历(针对每一个类别进行遍历)
            for class_id in range(1, per_img_scores.size(1)):  # skip background
                #TODO 得到当前的检测类别scores
                scores = per_img_scores[:, class_id]
                #TODO 使用阈值进行初筛
                mask = scores > self.cfg.TEST.CONFIDENCE_THRESHOLD
                #TODO 得到初筛之后的scores
                scores = scores[mask]
                if scores.size(0) == 0:
                    continue
                #TODO 得到初筛之后的boxes
                boxes = per_img_boxes[mask, :]
                #TODO 将其boxes的值还原回原图大小
                boxes[:, 0::2] *= self.width
                boxes[:, 1::2] *= self.height

                if self.is_soft_nms == False:
                    #TODO 进行NMS处理
                    keep = boxes_nms(boxes, scores,
                                     self.cfg.TEST.NMS_THRESHOLD,
                                     self.cfg.TEST.MAX_PER_CLASS)
                    #TODO 得到NMS之后的boxes
                    nmsed_boxes = boxes[keep, :]
                    #TODO 生成NMS之后类别标签
                    nmsed_labels = torch.tensor([class_id] * keep.size(0), device=device)
                    #TODO 得到对应的scores
                    nmsed_scores = scores[keep]

                else:
                    labels = torch.tensor([class_id] * boxes.size(0),device=device)
                    dets = soft_nms(boxes=boxes,scores=scores,labels=labels,
                                    conf_threshold=self.cfg.TEST.CONFIDENCE_THRESHOLD,
                                    nms_threshold=self.cfg.TEST.NMS_THRESHOLD,
                                    gamma = 0.5,
                                    soft_nms_fn='linear')
                    if len(dets) == 0:
                        continue
                    # TODO 得到NMS之后的boxes
                    nmsed_boxes = dets[:,:4]
                    # TODO 生成NMS之后类别标签
                    nmsed_labels = dets[:,5]
                    # TODO 得到对应的scores
                    nmsed_scores = dets[:,4]

                processed_boxes.append(nmsed_boxes)
                processed_scores.append(nmsed_scores)
                processed_labels.append(nmsed_labels)

            if len(processed_boxes) == 0:
                processed_boxes = torch.empty(0, 4)
                processed_labels = torch.empty(0)
                processed_scores = torch.empty(0)
            else:
                processed_boxes = torch.cat(processed_boxes, 0)
                processed_labels = torch.cat(processed_labels, 0)
                processed_scores = torch.cat(processed_scores, 0)

            #TODO 只要求取值前K个scores最大的
            if processed_boxes.size(0) > self.cfg.TEST.MAX_PER_IMAGE > 0:
                processed_scores, keep = torch.topk(processed_scores,
                                                    k=self.cfg.TEST.MAX_PER_IMAGE)
                processed_boxes = processed_boxes[keep, :]
                processed_labels = processed_labels[keep]

            #TODO 将所有的值拼接在一起
            container = Container(boxes=processed_boxes, labels=processed_labels,
                                  scores=processed_scores)
            container.img_width = self.width
            container.img_height = self.height
            results.append(container)
        return results
