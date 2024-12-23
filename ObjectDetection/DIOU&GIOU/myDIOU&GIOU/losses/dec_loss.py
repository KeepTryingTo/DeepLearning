import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.autograd import Variable

#############################################################################################
# encode
def encode(match_boxes, priors, variances):
    #TODO 计算匹配的gt box和anchor box之间的中心偏移
    c_yx = (match_boxes[:,:2]+match_boxes[:,2:]).float()/2-priors[:,:2]
    c_yx = c_yx.float()/(variances[0]*priors[:,2:])
    #TODO 计算高宽比
    hw = (match_boxes[:,2:]-match_boxes[:,:2]).float()/priors[:,2:]
    hw = torch.log(hw.float())/variances[1]

    return torch.cat([c_yx, hw], dim=1)


#############################################################################################
# match
def split_to_box(priors):
    #TODO 将anchor box[cx,cy,w,h] => [xmin,ymin,xmax,ymax]
    return torch.cat([priors[:,:2]-priors[:,2:]/2,
                      priors[:,:2]+priors[:,2:]/2],
                     1)

#TODO 计算交集area
def intersect(boxes_a, boxes_b):
    num_a = boxes_a.size(0)
    num_b = boxes_b.size(0)
    max_xy = torch.min(boxes_a[:,2:].unsqueeze(1).expand(num_a,num_b,2),
                       boxes_b[:,2:].unsqueeze(0).expand(num_a,num_b,2))
    min_xy = torch.max(boxes_a[:,:2].unsqueeze(1).expand(num_a,num_b,2),
                       boxes_b[:,:2].unsqueeze(0).expand(num_a,num_b,2))

    inter = torch.clamp((max_xy-min_xy), min=0.)
    return inter[:,:,0]*inter[:,:,1]


def jaccard(boxes_a, boxes_b):
    inter = intersect(boxes_a, boxes_b)
    #TODO 计算并集
    area_a = ((boxes_a[:,2]-boxes_a[::,0])*(boxes_a[:,3]-boxes_a[:,1])).unsqueeze(1).expand_as(inter)
    area_b = ((boxes_b[:,2]-boxes_b[::,0])*(boxes_b[:,3]-boxes_b[:,1])).unsqueeze(0).expand_as(inter)
    union = area_a+area_b-inter
    #TODO 计算IOU
    return inter/union



def match(gt_boxes, gt_label,
          priors, match_thresh,
          variances):
    # gt_boxes: y1,x1,y2,x2
    # priors: cy,cx,h,w
    # transfer to y1,x1,y2,x2
    priors_box = split_to_box(priors)
    overlaps = jaccard(gt_boxes, priors_box) # [num_gt, num_priors, 1]
    #TODO 计算和anchor box的IOU最大的那些gt box
    best_gt, best_gt_idx = overlaps.max(0, keepdim=True)
    best_gt.squeeze_(0)
    best_gt_idx.squeeze_(0)
    # TODO 计算和gt box的IOU最大的那些anchor box
    best_prior, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior.squeeze_(1)
    best_prior_idx.squeeze_(1)
    #TODO 将gt box和anchor box最有可能匹配的那些anchor索引位置的IOU设置为2
    best_gt.index_fill_(0, best_prior_idx, 2)
    #TODO 将best_gt_idx中的索引使用0~len(gt_box) - 1之间的值来表示
    for j in range(best_prior_idx.size(0)): #iterate num_a
        best_gt_idx[best_prior_idx[j]] = j
    #TODO 得到和anchor box最佳匹配的gt box
    match_boxes = gt_boxes[best_gt_idx]
    match_label = gt_label[best_gt_idx]
    #TODO 过滤掉可能得背景
    match_label[best_gt<match_thresh] = 0.
    encoded_boxes = encode(match_boxes,priors,variances)

    # print('encoded_boxes.shape',encoded_boxes.size())
    # print('match_label.shape',match_label.size())
    return encoded_boxes, match_label


#############################################################################################

class DEC_loss(Module):
    def __init__(self, num_classes, variances,
                 device, match_thresh=0.5,
                 neg_pos_ratio=3):
        super(DEC_loss, self).__init__()
        self.num_classes = num_classes
        self.variances = variances
        self.device = device
        self.match_thresh = match_thresh
        self.neg_pos_ratio = neg_pos_ratio

    def log_sum_exp(self, x):
        """This will be used to determine un-averaged confidence losses across
        all examples in a batch.
        """
        # x: [-1, num_classes]
        x_max = x.data.max() # get the max value of all - > output one value
        return torch.log(
            torch.sum(
                torch.exp(x-x_max),
                dim=1, #TODO 对指定的维度，也就是num_priors维度求和
                keepdim=True
            )
        )+x_max


    def forward(self, predictions, targets):
        p_locs, p_conf, prior_boxes = predictions
        prior_boxes = prior_boxes[:p_locs.size(1),:]
        batch_size = p_locs.size(0)
        num_priors = prior_boxes.size(0)

        # encode the matched groundtruth...
        t_locs = torch.FloatTensor(batch_size, num_priors, 4)
        t_conf = torch.LongTensor(batch_size, num_priors)
        for idx in range(batch_size):
            t_boxes = targets[idx][:,:-1].data
            t_label = targets[idx][:,-1].data
            d_boxes = prior_boxes.data
            encoded_boxes, encoded_label = match(gt_boxes=t_boxes,
                                                 gt_label=t_label,
                                                 priors=d_boxes,
                                                 match_thresh=self.match_thresh,
                                                 variances=self.variances)
            t_locs[idx] = encoded_boxes
            t_conf[idx] = encoded_label
        #TODO t_locs中保存的是anchor box和gt box之间的偏移量以及高宽比
        t_locs = t_locs.to(self.device)
        t_conf = t_conf.to(self.device)

        #TODO 计算正样本的数量
        pos_mask = t_conf>0 # batch x num_box
        num_pos = pos_mask.long().sum(dim=1,keepdim=True)

        # locs losses
        pos_locs_mask = pos_mask.unsqueeze(2).expand_as(p_locs)
        #TODO 预测的框偏移量以及置信度和真实的gt box & anchor box之间偏移量的损失计算
        loss_locs = F.smooth_l1_loss(input = p_locs[pos_locs_mask].view(-1,4),
                                     target= t_locs[pos_locs_mask].view(-1,4),
                                     size_average=False)

        # TODO 计算置信度损失conf losses
        # TODO hard negtive mining - 难例挖掘负样本
        p_conf_batch = p_conf.view(-1, self.num_classes)
        #TODO gather用于从输入张量中根据指定索引抽取元素的函数，并且指定抽取元素的维度dim=1
        temp = self.log_sum_exp(p_conf_batch)-p_conf_batch.gather(
            dim=1, index=t_conf.view(-1,1)
        )
        temp = temp.view(batch_size, -1)
        #TODO 对于那些正样本的位置负值为0，表示不进行关注
        temp[pos_mask] = 0.
        #TODO 将计算的置信度损失按照降序排序
        _, temp_idx = temp.sort(1, descending=True)
        #TODO 对降序排序的索引按照升序排序
        _, idx_rank = temp_idx.sort(1)
        #TODO 负样本的数量，其中pos_mask.size(1)=num_priors
        num_neg = torch.clamp(self.neg_pos_ratio*num_pos, max=pos_mask.size(1)-1)
        neg_mask = idx_rank < num_neg.expand_as(idx_rank)

        # TODO conf losses calc
        pos_conf_mask = pos_mask.unsqueeze(2).expand_as(p_conf)
        neg_conf_mask = neg_mask.unsqueeze(2).expand_as(p_conf)
        #TODO 计算（负样本 + 正样本）的置信度损失
        loss_conf = F.cross_entropy(input=p_conf[
                                        (pos_conf_mask+neg_conf_mask).gt(0)
                                    ].view(-1,self.num_classes),
                                    target=t_conf[(pos_mask+neg_mask).gt(0)],
                                    size_average=False)
        N = num_pos.data.sum().float()
        if N == 0:
            N = 1.
        # print(loss_locs/N)
        # print(loss_conf/N)
        return loss_locs/N, loss_conf/N