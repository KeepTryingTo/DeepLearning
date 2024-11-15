import torch
from torch.autograd import Function
import torch.nn as nn
from ..box_utils import decode, center_size
from ..box_utils import nms
# from torchvision.ops import nms
from data import data_configs

cfg = data_configs['VOC']


def myNMS(boxes, scores,iou_threshold = 0.45,topk = 200):
    scores,idx = torch.sort(scores,descending=True)
    boxes = boxes[idx[:topk]]
    scores = scores[:topk]

    keep = nms(boxes=boxes,
               scores=scores,
               iou_threshold=iou_threshold)
    return keep,boxes,scores


class Detect_RefineDet(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, size,
                 bkg_label, top_k,
                 conf_thresh, nms_thresh,
                 objectness_thre, keep_top_k):
        #super(Detect_RefineDet, self).__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.objectness_thre = objectness_thre
        self.variance = cfg[str(size)]['variance']

    def __call__(self, arm_loc_data, arm_conf_data,
                 odm_loc_data, odm_conf_data,
                 prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        loc_data = odm_loc_data
        conf_data = odm_conf_data

        arm_object_conf = arm_conf_data.data[:, :, 1:]
        #TODO 首先初步过滤掉那些不可能存在物体的位置
        no_object_index = arm_object_conf <= self.objectness_thre
        conf_data[no_object_index.expand_as(conf_data)] = 0

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # TODO Decode predictions into bboxes.
        for i in range(num):
            #TODO 根据anchor box和ARM预测偏移量解码得到实际的预测结果
            prior = prior_data.to(loc_data.device)
            default = decode(loc=arm_loc_data[i], priors=prior,
                             variances=self.variance)
            #TODO [xmin,ymin,xmax,ymax] => [cx,cy,w,h]
            default = center_size(default)
            #TODO 实际预测框的解码
            decoded_boxes = decode(loc=loc_data[i], priors=default,
                                   variances=self.variance)
            # TODO For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                #TODO 进一步过滤掉那些小于指定置信度的位置
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]

                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)

                # TODO idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes=boxes, scores=scores,
                                 overlap=self.nms_thresh,
                                 top_k=self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                                        boxes[ids[:count]]), dim = 1)

                # TODO 对于当前预测的类别，过滤掉那些重叠的框
                # keep, boxes, scores = myNMS(boxes=boxes, scores=scores,
                #                             iou_threshold=self.nms_thresh, topk=200)
                # output[i, cl, :len(keep)] = torch.cat(
                #     (
                #         scores[keep].unsqueeze(1), boxes[keep]
                #     ),
                #     dim=1
                # )
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.keep_top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
