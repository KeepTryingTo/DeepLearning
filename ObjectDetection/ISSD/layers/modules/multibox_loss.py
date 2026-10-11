import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp



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


    def __init__(self, num_classes,
                 overlap_thresh = 0.5,
                 prior_for_matching = True,
                 bkg_label = 0,
                 neg_mining = True,
                 neg_pos = 3,
                 neg_overlap = 0.5,
                 encode_target = False,
                 device = 'cuda'):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching  = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1,0.2]
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
        #TODO 得到batch的大小
        num = loc_data.size(0)
        #TODO 得到对应anchor的数量
        num_priors = (priors.size(0))
        #TODO 预测类别数
        num_classes = self.num_classes

        # TODO 用于anchor和gt box之间的匹配 match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        #TODO 遍历当前的batch
        for idx in range(num):
            #TODO 得到box坐标框和对应的类别标签
            truths = targets[idx][:,:-1].data
            labels = targets[idx][:,-1].data
            #TODO 得到anchor
            defaults = priors.data
            #TODO anchor和gt box之间的匹配
            match(self.threshold,truths,defaults,
                  self.variance,labels,
                  loc_t,conf_t,idx)
        loc_t = loc_t.to(self.device)
        conf_t = conf_t.to(self.device)
        # TODO wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)

        #TODO 过滤掉哪些置信度为0的，获得正样本的数量
        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        #TODO 根据正样本索引分别获得对应的预测定位框和标注框,并基于smooth_l1_loss计算定位损失
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # TODO Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1,self.num_classes)
        #TODO 将预测的confidence通过softmax转换为概率之后
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(dim = 1, index = conf_t.view(-1,1))

        # TODO Hard Negative Mining
        loss_c[pos.view(-1,1)] = 0 # filter out pos boxes for now TODO 过滤掉正样本，更加的关注负样本
        loss_c = loss_c.view(num, -1)
        _,loss_idx = loss_c.sort(1, descending=True) #TODO 按照计算的差值进行降序排序，得到对应的索引
        _,idx_rank = loss_idx.sort(1) #TODO 按照索引的大小进行升序排序，
        num_pos = pos.long().sum(1,keepdim=True) #TODO 得到正样本的数量
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1) #TODO 根据负样本和正样本之间比得到负样本的数量
        neg = idx_rank < num_neg.expand_as(idx_rank) #TODO 只针对那些小于负样本数量的索引作为最终的负样本

        # TODO Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = max(num_pos.data.sum().float(), 1)
        loss_l/=N
        loss_c/=N
        return loss_l,loss_c
