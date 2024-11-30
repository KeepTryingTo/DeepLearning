"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/24 21:25
"""

import torch
from torch import nn
import numpy as np
from utiles.iou import box_iou
import torch.nn.functional as F
from utiles.meshgrid import meshgrid_xy

class Yolov3Loss(nn.Module):
    def __init__(
            self,img_size,S,B,num_classes,anchors,
            lambda_coord = 5.,lambda_noobj = .5,device = 'cpu',
            lambda_prior = 5.,lambda_obj = 5.,lambda_class = 5.,eps = 1e-6
    ):
        super(Yolov3Loss, self).__init__()
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
        for i in range(0, len(anchors)):
            an_wh = []
            for wh in anchors[i]:
                # 由于给定的anchor未进行归一化处理;将其anchor的大小缩放至[0,S]之间的大小
                anchor_wh = (wh[0] / img_size * S[i], wh[1] / img_size * S[i])
                an_wh.append(anchor_wh)
            self.anchors.append(an_wh)

    def decoder_loc(self,outputs,num_output):
        """
        output:
            [1,3,(5 + num_classes),13,13]
            [1,3,(5 + num_classes),26,26]
            [1,3,(5 + num_classes),52,52]
        """
        fmsize = []
        for i in range(num_output):
            fmsize.append(self.S[i])
        # 获得txty
        loc_txtys, loc_twths, xys, box_xys = [], [], [], []
        for i in range(num_output):
            loc_txtys.append(outputs[i][:,:, :2, :, :])
            loc_twths.append(outputs[i][:,:, 2:4, :, :])
            # 生成网格 [fmsize,fmsize,2] => [2,fmsize,fmsize]
            xys.append(meshgrid_xy(fmsize[i]).view(fmsize[i], fmsize[i], 2).permute(2, 0, 1).to(self.device))
            # [5,2,fmsize,fmsize]
            box_xys.append(loc_txtys[i].sigmoid() + xys[i].expand_as(loc_txtys[i]))

        # 转换anchor的类型为tensor
        box_whs = []
        for i in range(len(self.S)):
            anchors_wh = torch.tensor(self.anchors[i])
            anchors_wh = anchors_wh.view(1,3,2,1,1).expand_as(loc_twths[i]).to(self.device)
            box_whs.append(loc_twths[i].exp() * anchors_wh)

        # 根据中心坐标和高宽转换为[xmin,ymin,xmax,ymax] => [5,4,fmsize,fmsize]
        boxes = []
        for i in range(num_output):
            box = torch.cat([box_xys[i] - box_whs[i] / 2, box_xys[i] + box_whs[i] / 2], dim=2)
            # box = box.permute(0, 2, 3, 1).view(-1, 4)
            boxes.append(box)

        return boxes

    def forward(self,output,loc_targets,cls_targets,boxes_target):
        """
        output:
            [1,(5 + num_classes)*3,13,13]
            [1,(5 + num_classes)*3,26,26]
            [1,(5 + num_classes)*3,52,52]
        loc_targets:[b,3,5,4,13,13]
        cls_targets:[b,3,5,num_classes,13,13]
        boxes_target: [b,num_boxes,4]
        """
        fmsize = []
        num_output = len(output)
        b,_,_,_ = output[0].size()
        for i in range(num_output):
            fmsize.append(output[i].size()[2])
        # TODO [1,125,13,13] => [5,25,13,13] == [5;(tx,ty,tw,th,c) + num_classes;13;13]
        outputs = []
        for i in range(num_output):
            outputs.append(output[i].view(b,3, 5 + self.num_classes,fmsize[i], fmsize[i]))
            # print('outputs[{}]: {}'.format(i,outputs[i]))
            # 获得txty
        loc_txtys, loc_twths = [], []
        for i in range(num_output):
            loc_txtys.append(outputs[i][:,:, :2, :, :])
            loc_twths.append(outputs[i][:,:, 2:4, :, :])

        # TODO 根据中心坐标和高宽转换为[xmin,ymin,xmax,ymax] => [5,4,fmsize,fmsize]
        loc_preds = []
        for i in range(num_output):
            box = torch.cat([loc_txtys[i], loc_twths[i]], dim=2)
            loc_preds.append(box)
        # TODO [3,b,4,fmsize,fmsize] => [b,3,4,fmsize,fmsize]
        loc_preds_t = []
        for i in range(b):
            loc_pred_t = []
            for k in range(num_output):
                loc_pred_t.append(loc_preds[k][i])
            loc_preds_t.append(loc_pred_t)
        #TODO 根据cls_targets得到在网格上包含的物体mask
        poss,masks = [],[]
        num_poss = 0.
        for i in range(b):
            cls_t = []
            for k in range(num_output):
                cls_t.append(cls_targets[i][k].max(dim = 1)[0].squeeze() > 0)
            poss.append(cls_t)
            num_poss += poss[i][0].data.long().sum()
            mask_t = []
            for k in range(num_output):
                mask_t.append(poss[i][k].unsqueeze(dim = 1).expand_as(loc_preds_t[i][k]))
            masks.append(mask_t)

        #TODO 计算定位误差
        loss_loc = 0.
        for i in range(b):
            for k in range(num_output):
                loss_loc += self.mse(
                    loc_preds_t[i][k][masks[i][k]], loc_targets[i][k][masks[i][k]]
                )

        #TODO 得到预测的confidence
        conf_preds = []
        for i in range(num_output):
            conf_preds.append(outputs[i][:,:, 4, :, :].sigmoid())
        # TODO [3,b,4,fmsize,fmsize] => [b,3,4,fmsize,fmsize]
        conf_preds_t = []
        for i in range(b):
            conf_pred_t = []
            for k in range(num_output):
                conf_pred_t.append(conf_preds[k][i])
            conf_preds_t.append(conf_pred_t)
        conf_targets = []
        for i in range(b):
            conf_target = []
            for k in range(num_output):
                conf_target.append(torch.zeros(conf_preds_t[i][k].size()).to(self.device))
            conf_targets.append(conf_target)
        #TODO 得到预测的boxes [b,3,4,fmsize,fmsize] => [b,3,fmsize,fmsize,4]
        boxes_preds = self.decoder_loc(outputs,num_output)
        boxes_preds_t = []
        for i in range(b):
            boxes_pred = []
            for k in range(num_output):
                boxes_pred.append(boxes_preds[k][i].permute(0,2,3,1).contiguous().view(-1,4))
            boxes_preds_t.append(boxes_pred)
        for i in range(b):
            #遍历一个batch；计算预测框和target框之间的iou
            for k in range(num_output):
                box_pred = boxes_preds_t[i][k]
                box_target = boxes_target[i][k]
                #计算预测框和真是框之间的iou
                ious = box_iou(box_pred,box_target)
                #根据计算的ious值，并且将预测框box和target box之间最大的iou作为target框的confidence
                conf_targets[i][k] = ious.max(dim = 1)[0].view(3,fmsize[k],fmsize[k])

        masks = []
        for i in range(b):
            mask = []
            for k in range(num_output):
                mask.append(torch.ones(size = conf_preds_t[i][k].size()).to(self.device) * 0.1)
                mask[k][poss[i][k]] = 1
            masks.append(mask)

        #计算置信度损失值
        loss_conf = 0.
        for i in range(b):
            for k in range(num_output):
                loss_conf += self.mse(
                    conf_preds_t[i][k] * masks[i][k],conf_targets[i][k] * masks[i][k]
                )
        #[b,5,num_classes,13,13] => [b,5,13,13,num_classes]
        cls_preds = []
        for i in range(b):
            cls_pred = []
            for k in range(num_output):
                cls_ = outputs[k][i][:,5:, :, :].permute(0,2,3, 1).contiguous().view(-1, self.num_classes)
                cls_ = F.softmax(cls_,dim = 1)
                cls_ = cls_.contiguous().view(3,fmsize[k],fmsize[k],self.num_classes).permute(0,3,1,2)
                cls_pred.append(cls_)
            cls_preds.append(cls_pred)
        loss_cls = 0.
        #获得有物体的位置
        for i in range(b):
            for k in range(num_output):
                pos = cls_targets[i][k] > 0
                #计算类别损失
                loss_cls += self.mse(
                    cls_preds[i][k][pos],cls_targets[i][k][pos]
                )
        # print('loc_loss: {}'.format(loss_loc))
        # print('conf_loss: {}'.format(loss_conf))
        # print('cls_loss: {}'.format(loss_cls))
        loss = self.lambda_coord * loss_loc + self.lambda_prior * loss_conf + self.lambda_class * loss_cls
        return loss / b / num_poss
