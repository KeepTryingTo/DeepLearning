"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/24 19:21
"""

import torch
from torch import nn
from utiles.meshgrid import meshgrid_xy
from utiles.iou import intersection_over_union,box_iou

class Encoder:
    def __init__(self,anchors,img_size,S = 13,B = 5,num_classes = 20):
        self.anchors = []
        for i in range(0,len(anchors),2):
            # 由于给定的anchor未进行归一化处理;将其anchor的大小缩放至[0,S]之间的大小
            anchor_wh = (anchors[i] / img_size * S,anchors[i + 1] / img_size * S)
            self.anchors.append(anchor_wh)
        self.img_size = img_size
        self.S = S
        self.B = B
        self.num_classes = num_classes

    def encoder(self,boxes,labels):
        """
         Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax) in range [0,1], sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int) model input size.

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [5,4,fmsize,fmsize] = [batch_size,4,fmap_size,fmap_size].
          cls_targets: (tensor) encoded class labels, sized [5,20,fmsize,fmsize] = [batch_size,20,fmap_size,fmap_size].
          box_targets: (tensor) truth boxes, sized [#obj,4].
        """
        num_boxes = len(boxes)
        # 320->10, 352->11, 384->12, 416->13, ..., 608->19
        fmsize = int((self.img_size - 320) / 32 + 10)
        #根据计算的最后输出特征图大小得到下采样的倍数
        gridSize = self.img_size / fmsize

        #根据boxes归一化之后，映射到指定的img size=416大小
        boxes *= self.img_size
        #获得中心坐标以及box的高宽，并且将其映射到[0,fmsize]之间的网格
        bx = ((boxes[...,0] + boxes[...,2]) / 2) / gridSize
        by = ((boxes[...,1] + boxes[...,3]) / 2) / gridSize
        bw = (boxes[...,2] - boxes[...,0]) / gridSize
        bh = (boxes[...,3] - boxes[...,1]) / gridSize

        #在fmsize网格中得到相对于左上角的网格坐标
        tx = bx - bx.floor()
        ty = by - by.floor()

        #生成一个fmsize * fmsize大小的网格
        xy = meshgrid_xy(fmsize) + 0.5 #offset the grid center
        xy = xy.view(fmsize, fmsize, 1, 2).expand(fmsize, fmsize, self.B, 2)

        #转换anchor的类型为tensor
        anchors_wh = torch.Tensor(self.anchors)
        anchors_wh = anchors_wh.view(1,1,self.B,2).expand(fmsize,fmsize,self.B,2)
        #将anchor_wh放到生成的网格上，表示每一个grid cell有self.B个anchor;注意是anchor的中心对应网格的中心
        #[fmsize,fmsize,5,4]；其中4 = [(xmin,ymin),(xmax,ymax)]
        anchor_boxes = torch.cat([xy - anchors_wh / 2,xy + anchors_wh / 2],dim = 3)

        #计算grid cell下的物体和每一个anchor求解iou，iou最大的，则由相应的anchor负责去回归
        ious = box_iou(anchor_boxes.view(-1,4),boxes / gridSize)
        #[N,M] => [fmsize,fmsize,self.B,num_boxes]
        ious = ious.view(fmsize,fmsize,self.B,num_boxes)

        #初始化loc_targets和cls_targets用于保存匹配的anchor和label
        loc_targets = torch.zeros(size=(self.B,4,fmsize,fmsize))
        cls_targets = torch.zeros(size=(self.B,self.num_classes,fmsize,fmsize))
        #根据计算的iou，来找到和object之间iou最大的anchor用于负责回归
        for i in range(num_boxes):
            #得到相对于网格的左上角坐标
            left_x ,top_y = int(bx[i]),int(by[i])
            #得到当前网格的(left_x ,top_y)匹配最大iou的anchor索引号
            max_iou,max_index = torch.max(ious[top_y,left_x,:,i],dim = 0)
            j = max_index.item()
            cls_targets[j,labels[i],top_y,left_x] = 1
            #计算当前box的宽高对应iou最大的anchor宽高之间比值
            tw,th = torch.log( bw[i] / self.anchors[j][0]),torch.log(bh[i] / self.anchors[j][1])
            #保存当前的宽高比以及左上角的left_x ,top_y
            loc_targets[j,:,top_y,left_x] = torch.Tensor([tx[i],ty[i],tw,th])
        return loc_targets,cls_targets,boxes / gridSize
    def decoder(self,output):
        """
        output: [1,125,13,13]
        """
        fmsize = output.size()[2]
        #[1,125,13,13] => [5,25,13,13] == [5;(tx,ty,tw,th,c) + num_classes;13;13]
        output = output.view(5,5 + self.num_classes,self.S,self.S)

        #获得txty
        loc_txtys = output[:,:2,:,:]
        #获得tw,th
        loc_twths = output[:,2:4,:,:]
        #获得confience
        confidences = output[:,4,:,:]
        #获得probility
        probs = output[:,5:,:,:]

        #生成网格 [fmsize,fmsize,2] => [2,fmsize,fmsize]
        xy = meshgrid_xy(fmsize).view(fmsize,fmsize,2).permute(2,0,1)
        #[5,2,fmsize,fmsize]
        box_xy = loc_txtys.sigmoid() + xy.expand_as(loc_txtys)

        #生成相应的anchors
        # 转换anchor的类型为tensor
        anchors_wh = torch.tensor(self.anchors)
        anchors_wh = anchors_wh.view(self.B, 2,1,1).expand_as(loc_twths)
        #[5,2,fmsize,fmsize]
        box_wh = loc_twths.exp() * anchors_wh

        #根据中心坐标和高宽转换为[xmin,ymin,xmax,ymax] => [5,4,fmsize,fmsize]
        boxes = torch.cat([box_xy - box_wh / 2,box_xy + box_wh / 2],dim = 1)
        boxes = boxes.permute(0,2,3,1).contiguous().view(-1,4)
        #[5,13,13] => [5 * 13 * 13]
        confidence = confidences.sigmoid().view(-1)
        #[5,20,13,13] => [5,13,13,20] => [5 * 13 * 13,20]
        cls_preds = probs.permute(0,2,3,1).contiguous().view(-1,self.num_classes)
        #[5 * 13 * 13,20]
        cls_preds = torch.softmax(cls_preds,dim=1)
        scores = cls_preds * confidence.unsqueeze(dim = 1).expand_as(cls_preds)
        #[5*13*13,20] => [5*13*13]
        scores = scores.max(dim = 1)[0].view(-1)
        #[5*13*13]
        cls_lables = cls_preds.max(dim = 1)[1].view(-1)
        return boxes / fmsize,scores,cls_lables

