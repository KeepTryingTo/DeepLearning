"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/10/29-12:30
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from utiles import box_utils
import torch.nn.functional as F
from utiles.loss import MultiBoxLoss
from models.box_predictor import BoxPredictor
from models.inference import PostProcessor
from models.prior_box import PriorBox

class DSSDBoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = BoxPredictor(cfg)
        #NEG_POS_RATIO： MODEL.NEG_POS_RATIO = 3（# Hard negative mining） 正负样本之比为3:1
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)
        self.priors = None

    def forward(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)
        # print('cls_logits: {}'.format(cls_logits.size()))
        # print('box_logits: {}'.format(bbox_pred.size()))
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        #得到GT的image和labels = [xmin,ymin,xmax,ymax,class_id]
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        #损失值计算
        reg_loss, cls_loss = self.loss_evaluator(cls_logits, bbox_pred, gt_labels, gt_boxes)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        detections = (cls_logits, bbox_pred)
        return detections, loss_dict

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        """
            CENTER_VARIANCE: 0.1 
            SIZE_VARIANCE: 0.2
        """
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE,
            self.cfg.MODEL.SIZE_VARIANCE
        )
        #TODO 将box的[x,y,w,h] => [xmin,ymin,xmax,ymax]
        boxes = box_utils.center_form_to_corner_form(boxes)
        #TODO 得到检测结果
        detections = (scores, boxes)
        #TODO 进行NMS处理,同时也将框转换为相对原图大小
        detections = self.post_processor(detections)
        #TODO 返回经过后处理之后的预测box和confidence，以及未经过预处理的box和conf
        return detections, cls_logits,bbox_pred