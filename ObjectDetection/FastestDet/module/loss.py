import math
import torch
import torch.nn as nn

class DetectorLoss(nn.Module):
    def __init__(self, device):    
        super(DetectorLoss, self).__init__()
        self.device = device

    def bbox_iou(self, box1, box2, eps=1e-7):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box1 = box1.t()
        box2 = box2.t()
        #TODO [cx,cy,h,w] => [xmin,ymin,xmax,ymax]
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # TODO 计算交集 Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union
        #TODO 计算预测box和真实box之间的最小外接矩形，
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        # TODO SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        #TODO 计算中心距离
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
        #TODO 计算角度的正弦值
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        #TODO sqrt(2) / 2的正弦或者预先值就是π/4
        threshold = pow(2, 0.5) / 2
        #TODO 得到α值，其中α为小于π/4的角度值
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        #TODO 1 - 2 * sin^2(arcsin(x) - π / 4) = cos(arcsin(x)^2 - π / 2)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

        #TODO 求解距离成本
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        #TODO 求解形状成本
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = (torch.pow(1 - torch.exp(-1 * omiga_w), 4) +
                      torch.pow(1 - torch.exp(-1 * omiga_h), 4))
        iou = iou - 0.5 * (distance_cost + shape_cost)

        return iou
        
    def build_target(self, preds, targets):
        N, C, H, W = preds.shape
        # TODO batch存在标注的数据
        gt_box, gt_cls, ps_index = [], [], []
        # todo 每个网格的四个顶点为box中心点会归的基准点
        quadrant = torch.tensor([[0, 0], [1, 0], 
                                 [0, 1], [1, 1]], device=self.device)

        if targets.shape[0] > 0:
            # todo 将坐标映射到特征图尺度上
            scale = torch.ones(6).to(self.device)
            scale[2:] = torch.tensor(preds.shape)[[3, 2, 3, 2]]
            #TODO 根据当前预测输出的特征图尺寸对target进行缩放
            gt = targets * scale

            # todo 扩展维度复制数据
            gt = gt.repeat(4, 1, 1)

            # todo 过滤越界坐标
            quadrant = quadrant.repeat(gt.size(1), 1, 1).permute(1, 0, 2)
            #TODO 其实这里已经蕴含了后面在计算回归损失时的中心偏移回归损失计算
            gij = gt[..., 2:4].long() + quadrant
            #TODO 边界处理
            j = torch.where(gij < H, gij, 0).min(dim=-1)[0] > 0 

            # todo 前景的位置下标
            gi, gj = gij[j].T
            batch_index = gt[..., 0].long()[j]
            ps_index.append((batch_index, gi, gj))

            # todo 前景的box
            gbox = gt[..., 2:][j]
            gt_box.append(gbox)
            
            # 前景的类别
            gt_cls.append(gt[..., 1].long()[j])

        return gt_box, gt_cls, ps_index

        
    def forward(self, preds, targets):
        # 初始化loss值
        ft = torch.cuda.FloatTensor if preds[0].is_cuda else torch.Tensor
        cls_loss, iou_loss, obj_loss = ft([0]), ft([0]), ft([0])

        # TODO 定义obj和cls的负对数似然损失函数，NLLLoss 是与 Softmax 激活函数配合使用的
        BCEcls = nn.NLLLoss() 
        # TODO smmoth L1相比于bce效果最好
        BCEobj = nn.SmoothL1Loss(reduction='none')
        
        # todo 构建ground truth
        gt_box, gt_cls, ps_index = self.build_target(preds, targets)

        pred = preds.permute(0, 2, 3, 1)
        # TODO 前背景分类分支
        pobj = pred[:, :, :, 0]
        # TODO 检测框回归分支
        preg = pred[:, :, :, 1:5]
        # TODO 目标类别分类分支
        pcls = pred[:, :, :, 5:]

        N, H, W, C = pred.shape
        tobj = torch.zeros_like(pobj) 
        factor = torch.ones_like(pobj) * 0.75

        if len(gt_box) > 0:
            # TODO 计算检测框回归loss
            b, gx, gy = ps_index[0]
            #todo 根据当前gt box的真实坐标来对相应预测位置的框坐标以及高宽进行解码
            ptbox = torch.ones((preg[b, gy, gx].shape)).to(self.device)
            ptbox[:, 0] = preg[b, gy, gx][:, 0].tanh() + gx
            ptbox[:, 1] = preg[b, gy, gx][:, 1].tanh() + gy
            ptbox[:, 2] = preg[b, gy, gx][:, 2].sigmoid() * W
            ptbox[:, 3] = preg[b, gy, gx][:, 3].sigmoid() * H

            # TODO 计算检测框IOU loss
            iou = self.bbox_iou(ptbox, gt_box[0])
            # Filter
            f = iou > iou.mean()
            b, gy, gx = b[f], gy[f], gx[f]

            # todo 计算iou loss
            iou = iou[f]
            iou_loss =  (1.0 - iou).mean() 

            # todo 计算目标类别分类分支loss
            ps = torch.log(pcls[b, gy, gx])
            cls_loss = BCEcls(ps, gt_cls[0][f])

            # TODO iou aware
            tobj[b, gy, gx] = iou.float()
            # TODO 统计每个图片正样本的数量
            n = torch.bincount(b)
            factor[b, gy, gx] =  (1. / (n[b] / (H * W))) * 0.25

        # 计算前背景分类分支loss
        obj_loss = (BCEobj(pobj, tobj) * factor).mean()

        # 计算总loss
        loss = (iou_loss * 8) + (obj_loss * 16) + cls_loss                      
              
        return iou_loss, obj_loss, cls_loss, loss