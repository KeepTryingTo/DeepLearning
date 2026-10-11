# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import torch
from utilities import box_utils
import numpy as np

class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        #TODO 将anchor box从[cx,cy,w,h] => [xmin,ymin,xmax,ymax]
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        #TODO 计算gt box和anchor box之间的匹配情况
        boxes, labels = box_utils.assign_priors(gt_boxes=gt_boxes,
                                                gt_labels=gt_labels,
                                                corner_form_priors=self.corner_form_priors,
                                                iou_threshold=self.iou_threshold)
        #TODO 将 匹配的gt box从[xmin,ymin,xmax,ymax] => [cx,cy,w,h]
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(center_form_boxes=boxes,
                                                         center_form_priors=self.center_form_priors,
                                                         center_variance=self.center_variance,
                                                         size_variance=self.size_variance)
        return locations, labels