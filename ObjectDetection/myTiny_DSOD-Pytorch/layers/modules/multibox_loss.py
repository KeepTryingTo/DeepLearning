#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
GPU = False
if torch.cuda.is_available():
    GPU = True


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh,
                 prior_for_matching, bkg_label,
                 neg_mining, neg_pos,
                 neg_overlap, encode_target,
                 device):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.device = device

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults,
                  self.variance, labels, loc_t, conf_t, idx)
        if GPU:
            loc_t = loc_t.to(self.device)
            conf_t = conf_t.to(self.device)
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - \
            batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)
                           ].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c

def GtAnchorMatch():
    overlaps = torch.tensor([
        [0.75, 0.11, 0.32, 0.01, 0.30, 0.20],
        [0.30, 0.65, 0.20, 0.25, 0.11, 0.23],
        [0.22, 0.45, 0.10, 0.31, 0.50, 0.30]
    ])
    truths = torch.tensor([1, 2, 3], dtype=torch.int)

    print('overlaps.shape: ', overlaps.size())

    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior  TODO 在A的维度上找到和B(anchor)最佳匹配的gt box坐标框
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    # TODO [1,n] => [n]
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    # TODO [m,1] => [m]
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    print('before best_prior_idx: {}'.format(best_prior_idx))
    print('before best_truth_idx: {}'.format(best_truth_idx))

    # TODO (dim = 0,index = best_prior_idx,value = 2)
    # TODO 将anchor和gt box匹配的最佳位置的IOU值都设置为2
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior

    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    # TODO 确保每个真实框都有其重叠最大的prior
    # TODO 同时将anchor和gt box匹配最佳，并且在best_truth_idx中的都设置为j
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j  # TODO 更新真实框索引，确保每个prior都有最佳匹配的真实框

    print('after best_truth_idx: {}'.format(best_truth_idx))
    print('after best_truth_overlap: {}'.format(best_truth_overlap))
    # TODO 根据最佳匹配的anchor
    matches = truths[best_truth_idx]  # Shape: [num_priors,4]
    print('matches: {}'.format(matches))

    # TODO 过滤掉那些低置信度的box和labels
    # conf[best_truth_overlap < threshold] = 0  # label as background

if __name__ == '__main__':
    GtAnchorMatch()
    pass