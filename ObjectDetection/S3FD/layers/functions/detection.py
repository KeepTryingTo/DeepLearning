
import torch
from ..bbox_utils import decode
# from ..bbox_utils import nms
from torch.autograd import Function
from torchvision.ops import nms

def myNMS(boxes, scores,iou_threshold = 0.45,topk = 200):
    #TODO 选择那些大于指定IOU并且位于前topk个坐标框
    scores,idx = torch.sort(scores,descending=True)
    boxes = boxes[idx[:topk]]
    scores = scores[:topk]

    keep = nms(boxes=boxes,
               scores=scores,
               iou_threshold=iou_threshold)
    return keep,boxes,scores


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg):
        self.num_classes = cfg.NUM_CLASSES
        self.top_k = cfg.TOP_K
        self.nms_thresh = cfg.NMS_THRESH
        self.conf_thresh = cfg.CONF_THRESH
        self.variance = cfg.VARIANCE
        self.nms_top_k = cfg.NMS_TOP_K

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4] 
        """
        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        conf_preds = conf_data.view(
            num, num_priors,
            self.num_classes
        ).transpose(2, 1)

        batch_priors = prior_data.view(-1, num_priors,
                                       4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        batch_priors = batch_priors.to(loc_data.device)
        #TODO 通过anchor box对预测框进行解码
        decoded_boxes = decode(loc_data.view(-1, 4),
                               batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output = torch.zeros(num, self.num_classes, self.top_k, 5)

        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()
            #TODO 针对每一个类别进行过滤
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                # ids, count = nms(
                #     boxes_, scores, self.nms_thresh, self.nms_top_k)
                # count = count if count < self.top_k else self.top_k
                #
                # output[i, cl, :count] = torch.cat(
                #     (scores[ids[:count]].unsqueeze(1),
                #             boxes_[ids[:count]]),
                #         dim = 1
                # )
                # TODO 对于当前预测的类别，过滤掉那些重叠的框
                keep, boxes, scores = myNMS(boxes=boxes_, scores=scores,
                                            iou_threshold=self.nms_thresh,
                                            topk=self.top_k)
                output[i, cl, :len(keep)] = torch.cat(
                    (
                        scores[keep].unsqueeze(1), boxes[keep]
                    ),
                    dim=1
                )

        return output
