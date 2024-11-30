"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/24 21:25
"""

import torch
from torch import nn
from utiles.iou import box_iou
import torch.nn.functional as F
from utiles.meshgrid import meshgrid_xy

class Yolov2Loss(nn.Module):
    def __init__(
            self,img_size,S,B,num_classes,anchors,
            lambda_coord = 5.,lambda_noobj = .5,device = 'cpu',
            lambda_prior = 5.,lambda_obj = 5.,lambda_class = 5.,eps = 1e-6
    ):
        super(Yolov2Loss, self).__init__()
        self.img_size = img_size
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.eps = eps
        self.device = device
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_prior = lambda_prior
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class
        self.mse = nn.MSELoss(reduction='sum')
        self.anchors = []
        for i in range(0,len(anchors),2):
            # TODO 由于给定的anchor未进行归一化处理;将其anchor的大小缩放至[0,S]之间的大小
            anchor_wh = (anchors[i] / img_size * S, anchors[i + 1] / img_size * S)
            self.anchors.append(anchor_wh)

    def decoder_loc(self,output):
        """
        output: [b,5,25,13,13]
        """
        b,_,_,fmsize,_ = output.size()
        #TODO 获得预测的定位中心坐标
        pred_xy = output[:,:,:2,:,:]
        #TODO 根据对应输出的特征图大小生成网格
        xy = meshgrid_xy(fmsize).view(fmsize,fmsize,2).permute(2,0,1)
        xy = xy.to(self.device)
        #TODO 根据dx,dy = sigmoid(tx) + cx,sigmoid(ty) + cy
        box_xy = pred_xy.sigmoid() + xy.expand_as(pred_xy) #[B,5,2,13,13]

        #TODO 得到预测高宽
        pred_wh = output[:,:,2:4,:,:]
        anchors_wh = torch.Tensor(self.anchors).view(1,5,2,1,1).expand_as(pred_wh).to(self.device)
        box_wh = pred_wh.exp() * anchors_wh #[B,5,2,13,13]

        #TODO 将预测得到的框放置到13 x 13的网格上
        box_preds = torch.cat([
            box_xy - box_wh / 2,box_xy + box_wh / 2
        ],dim = 2)

        return box_preds

    def forward(self,output,loc_targets,cls_targets,boxes_target):
        """
        output:[b,125,13,13]
        loc_targets:[b,5,4,13,13]
        cls_targets:[b,5,num_classes,13,13]
        boxes_target: [b,num_boxes,4]
        """
        b,c,fmsize,fmsize = output.size()
        # [b,125,13,13] => [b,5,25,13,13] == [b,5;(tx,ty,tw,th,c) + num_classes;13;13]
        output = output.view(b,5, 5 + self.num_classes, self.S, self.S)
        # 获得txty [B,5,2,13,13]
        loc_txtys = output[:,:, :2, :, :].sigmoid()
        # 获得tw,th [B,5,2,13,13]
        loc_twths = output[:,:, 2:4, :, :]
        loc_preds = torch.cat([loc_txtys,loc_twths],dim = 2)

        #根据cls_targets得到在网格上包含的物体mask
        pos = cls_targets.max(dim = 2)[0].squeeze() > 0
        #计算包含物体的数量
        num_pos = pos.data.long().sum()
        #将pos维度扩展到loc_preds
        mask = pos.unsqueeze(dim = 2).expand_as(loc_preds)

        #计算定位误差
        # loss_loc = F.smooth_l1_loss(
        #     loc_preds[mask],loc_targets[mask],reduction='sum'
        # )
        loss_loc = self.mse(
            loc_preds*mask, loc_targets*mask
        )

        # print('loc_preds: {}'.format(loc_preds[mask]))
        # print('loc_target: {}'.format(loc_targets[mask]))

        #得到预测的confidence
        conf_preds = output[:,:,4,:,:].sigmoid()
        conf_target = torch.zeros(conf_preds.size()).to(self.device)
        #TODO 得到预测的boxes [b,5,4,13,13] => [b,5,13,13,4]
        boxes_preds = self.decoder_loc(output).permute(0,1,3,4,2).contiguous().view(b,-1,4)

        #遍历一个batch；计算预测框和target框之间的iou
        for i in range(b):
            box_pred = boxes_preds[i]
            box_target = boxes_target[i]
            #计算预测框和真是框之间的iou: [5*fmsize*fmsize,num_boxes]
            ious = box_iou(box_pred,box_target)
            #根据计算的ious值，并且将预测框box和target box之间最大的iou作为target框的confidence: [N,M]
            conf_target[i] = ious.max(dim = 1)[0].view(5,fmsize,fmsize)
        mask = (torch.ones(size = conf_preds.size()).to(self.device) * 0.1)
        mask[pos] = 1
        #计算置信度损失值
        # loss_conf = F.smooth_l1_loss(
        #     conf_preds*mask,conf_target*mask,reduction='sum'
        # )
        loss_conf = self.mse(
            conf_preds*mask, conf_target*mask
        )
        # print('conf_preds: {}'.format(conf_preds[pos]))
        # print('conf_target: {}'.format(conf_target[pos]))
        #[b,5,num_classes,13,13] => [b,5,13,13,num_classes]
        cls_preds = output[:,:,5:,:,:].permute(0,1,3,4,2).contiguous().view(-1,self.num_classes)
        cls_preds = F.softmax(cls_preds,dim = 1)
        cls_preds = cls_preds.view(b,5,fmsize,fmsize,self.num_classes).permute(0,1,4,2,3)
        #获得有物体的位置
        pos = cls_targets > 0
        #计算类别损失
        # loss_cls = F.smooth_l1_loss(
        #     cls_preds[pos],cls_targets[pos],reduction='sum'
        # )
        loss_cls = self.mse(
            cls_preds[pos], cls_targets[pos]
        )
        # print('cls_preds: {}'.format(cls_preds[pos]))
        # print('cls_target: {}'.format(cls_targets[pos]))
        # print('loss_loc: {}'.format(loss_loc))
        # print('loss_conf: {}'.format(loss_conf))
        # print('loss_cls: {}'.format(loss_cls))
        loss = self.lambda_coord * loss_loc + self.lambda_prior * loss_conf + self.lambda_class * loss_cls
        return loss / b
