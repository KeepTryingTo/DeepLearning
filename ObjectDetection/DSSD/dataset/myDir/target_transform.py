import numpy as np
import torch

from utiles import box_utils


class SSDTargetTransform:
    __annotations__ = {
        'center_form_priors':'prior anchors',
        'center_variance':0.1,
        'size_variance': 0.2,
        'iou_threshold':0.5
    }
    def __init__(self, center_form_priors,
                 center_variance,
                 size_variance,
                 iou_threshold):
        #TODO 得到prior anchors
        self.center_form_priors = center_form_priors
        #TODO [x,y,w,h] => [xmin,ymin,xmax,ymax]
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        #TODO 0.1
        self.center_variance = center_variance
        #TODO 0.2
        self.size_variance = size_variance
        #TODO IOU阈值为0.5
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        #TODO 为GT box匹配prior anchor box
        boxes, labels = box_utils.assign_priors(
            gt_boxes,
            gt_labels,
            self.corner_form_priors,
            self.iou_threshold
        )
        #TODO [xmin,ymin,xmax,ymax] => [x,y,w,h]
        boxes = box_utils.corner_form_to_center_form(boxes)
        #TODO 计算anchor box和gt box之间的中心偏移和高宽比
        locations = box_utils.convert_boxes_to_locations(
            boxes,
            self.center_form_priors,
            self.center_variance,
            self.size_variance
        )
       
        return locations, labels
