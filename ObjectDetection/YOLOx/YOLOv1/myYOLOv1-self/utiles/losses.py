"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/12 16:50
"""

import torch
from torch import nn
from utiles.iou import intersection_over_union

class Yolov1Loss(nn.Module):
    def __init__(
            self,lambda_noobj = .5,lambda_coord = 5.,
            num_classes = 20,B = 2,eps = 1e-6
    ):
        super(Yolov1Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord
        self.num_classes = num_classes
        self.B = B
        self.eps = eps

    def forward(self,output,gt_map):
        """
        :param output: darknet's output [B,7,7,num_classe + B * 5]
        :param gt_map: corrsponding to target [B,7,7,num_class + B * 5]
        :return: loss's value
        """
        batch_size = gt_map.size()[0]
        #TODO 得到对应预测的box坐标以及对应的confience
        pred_boxes1 = output[:,:,:,self.num_classes:self.num_classes + 4]
        pred_boxes2 = output[:,:,:,self.num_classes + 5:self.num_classes + 9]
        gt_boxes1 = gt_map[:,:,:,self.num_classes:self.num_classes + 4]
        gt_boxes2 = gt_map[:,:,:,self.num_classes + 5:self.num_classes + 9]
        #TODO 计算pred_boxes和gt_boxes每一个grid cell中的两个box之间对应IOU，然后在两个box之间选择IOU最大的作为负责
        # 预测目标的框 iou1 and iou2: [B,S,S,1]
        iou1 = intersection_over_union(pred_boxes1,gt_boxes1)
        iou2 = intersection_over_union(pred_boxes2,gt_boxes2)
        #TODO [2,B,S,S,1]
        iou = torch.cat([iou1.unsqueeze(dim = 0),iou2.unsqueeze(dim = 0)],dim=0)
        #TODO 计算其中iou最大的值以及对应的索引: value: [B,S,S,1]; index: [B,S,S,1]
        iou_max_value,iou_max_index = torch.max(iou,dim = 0)
        # TODO https://blog.csdn.net/yangyanbao8389/article/details/121477053
        # TODO 在维度为dim = -1，并且索引为index的位置，将其值重新分配到torch.zeros_like(p_template)中
        template= output[...,self.num_classes + 4::5] > 0
        index = torch.argmax(iou, dim=0, keepdim=True).squeeze(0)
        responsible = torch.zeros_like(template).scatter_(  # (batch, S, S, 1)
            dim=-1,
            index=index,  # (batch, S, S, 1)
            value=1  # 1 if bounding box is "responsible" for predicting the object
        )
        #TODO 根据gt_map中confidence的值，为1的位置表示有目标obj,而gt_map的confidence为0的地方则表示无目标noobj
        exist_obi = (gt_map[:,:,:,self.num_classes + 4]).unsqueeze(dim = 3)
        # [B,S,S,1] # 1 if object exists AND bbox is responsible,Otherwise, confidence should be 0
        exist_obij = exist_obi * responsible #[B,S,S,1]
        noobij = 1 - exist_obij #[B,S,S,1]
        #TODO 首先计算坐标之间的误差
        #output[...,self.num_classes::5]: [B,7,7,2]
        #mse(xi,xj) + mse(yi,yj)
        loss_coord = (
            self.mse(
                torch.flatten(exist_obij * output[...,self.num_classes::5]),
                torch.flatten(exist_obij * gt_map[...,self.num_classes::5])
            )
            + self.mse(
                torch.flatten(exist_obij * output[...,self.num_classes + 1::5]),
                torch.flatten(exist_obij * gt_map[...,self.num_classes + 1::5])
            )
        )

        #TODO 计算高宽之间的误差
        loss_wh = (
            self.mse(
                torch.flatten(exist_obij * torch.sqrt(torch.abs(output[..., self.num_classes + 2::5]) + self.eps)),
                torch.flatten(exist_obij * torch.sqrt(gt_map[..., self.num_classes + 2::5]) + self.eps)
            )
            + self.mse(
                torch.flatten(exist_obij * torch.sqrt(torch.abs(output[..., self.num_classes + 3::5]) + self.eps)),
                torch.flatten(exist_obij * torch.sqrt(gt_map[..., self.num_classes + 3::5]))
            )
        )
        #TODO 计算负责检测物体的confidence误差 Confidence losses (target confidence is IOU)
        loss_conf_obj = (
            self.mse(
                torch.flatten(exist_obij * output[..., self.num_classes + 4::5]),
                # torch.flatten(exist_obij * torch.ones_like(iou_max_value.repeat(1,1,1,2)))
                torch.flatten(exist_obij * gt_map[...,self.num_classes + 4::5])
            )
        )
        #TODO 计算不负责检测物体的confidence误差
        loss_conf_noobj = (
            self.mse(
                torch.flatten(noobij * output[..., self.num_classes + 4::5]),
                # torch.flatten(noobij * torch.zeros_like(iou_max_value.repeat(1,1,1,2)))
                torch.flatten(noobij * gt_map[...,self.num_classes + 4::5])
            )
        )
        #TODO 计算最后的预测概率误差
        # prob_p = output[...,:self.num_classes]
        # prob_t = gt_map[...,:self.num_classes]
        loss_prob = (
            self.mse(
                exist_obi * output[...,:self.num_classes],
                exist_obi * gt_map[...,:self.num_classes]
            )
        )
        loss = (
            self.lambda_coord*loss_coord
            + self.lambda_coord*loss_wh
            + loss_conf_obj
            + self.lambda_noobj*loss_conf_noobj
            + loss_prob
        ) / batch_size
        return loss
