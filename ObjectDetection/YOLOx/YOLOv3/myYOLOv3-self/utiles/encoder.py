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
    def __init__(self,anchors,img_size,S = (13,26,52),
                 B = 3,num_classes = 20,device = 'cpu'):
        self.anchors = []
        for i in range(0,len(anchors)):
            an_wh = []
            for wh in anchors[i]:
                # 由于给定的anchor未进行归一化处理;将其anchor的大小缩放至[0,S]之间的大小
                anchor_wh = (wh[0] / img_size * S[i],wh[1] / img_size * S[i])
                an_wh.append(anchor_wh)
            self.anchors.append(an_wh)
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
        fmsize = []
        for i  in range(len(self.S)):
            fmsize.append(int(self.S[i]))
        #TODO 根据计算的最后输出特征图大小得到下采样的倍数
        gridSize = []
        for i in range(len(self.S)):
            gridSize.append(self.img_size / fmsize[i])

        #TODO 根据boxes归一化之后，映射到指定的img size=416大小
        boxes *= self.img_size
        #TODO 获得中心坐标以及box的高宽，并且将其映射到[0,fmsize]之间的网格
        bxbybwbh = []
        for i in range(len(self.S)):
            bx = (((boxes[...,0] + boxes[...,2]) / 2) / gridSize[i]).unsqueeze(dim = 1)
            by = (((boxes[...,1] + boxes[...,3]) / 2) / gridSize[i]).unsqueeze(dim = 1)
            bw = (((boxes[...,2] - boxes[...,0])) / gridSize[i]).unsqueeze(dim = 1)
            bh = (((boxes[...,3] - boxes[...,1])) / gridSize[i]).unsqueeze(dim = 1)
            box = torch.cat([bx,by,bw,bh],dim = 1)
            bxbybwbh.append(box)

        #在fmsize网格中得到相对于左上角的网格坐标
        txty = []
        for i in range(len(self.S)):
            tx = (bxbybwbh[i][...,0] - bxbybwbh[i][...,0].floor()).unsqueeze(dim = 1)
            ty = (bxbybwbh[i][...,1] - bxbybwbh[i][...,1].floor()).unsqueeze(dim = 1)
            txty.append(torch.cat([tx,ty],dim = 1))

        #生成一个fmsize * fmsize大小的网格
        xys = []
        for i in range(len(self.S)):
            xy = meshgrid_xy(fmsize[i]) + 0.5 #offset the grid center
            xy = xy.view(fmsize[i], fmsize[i], 1, 2).expand(fmsize[i],
                                                            fmsize[i],
                                                            self.B, 2)
            xys.append(xy)

        #转换anchor的类型为tensor
        anchors_whs = []
        for i in range(len(self.S)):
            anchors_wh = torch.tensor(self.anchors[i])
            anchors_wh = anchors_wh.view(1,1,self.B,2).expand(fmsize[i],fmsize[i],self.B,2)
            anchors_whs.append(anchors_wh)
        #将anchor_wh放到生成的网格上，表示每一个grid cell有self.B个anchor;注意是anchor的中心对应网格的中心
        #[fmsize,fmsize,5,4]；其中4 = [(xmin,ymin),(xmax,ymax)]
        anchor_boxes = []
        for i in range(len(self.S)):
            anchor_box = torch.cat([xys[i] - anchors_whs[i] / 2,
                                    xys[i] + anchors_whs[i] / 2],dim = 3)
            anchor_boxes.append(anchor_box)

        #计算grid cell下的物体和每一个anchor求解iou，iou最大的，则由相应的anchor负责去回归
        ious = []
        for i in range(len(self.S)):
            iou = box_iou(anchor_boxes[i].view(-1,4),boxes / gridSize[i])
            # [N,M] => [fmsize,fmsize,self.B,num_boxes]
            iou = iou.view(fmsize[i], fmsize[i], self.B, num_boxes)
            ious.append(iou)
        #初始化loc_targets和cls_targets用于保存匹配的anchor和label
        loc_targets = []
        cls_targets = []
        for i in range(len(self.S)):
            loc_targets.append(torch.zeros(size=(self.B,4,fmsize[i],fmsize[i])))
            cls_targets.append(torch.zeros(size=(self.B,self.num_classes,fmsize[i],fmsize[i])))
        t_boxes = []
        for i in range(len(self.S)):
            #根据计算的iou，来找到和object之间iou最大的anchor用于负责回归
            for k in range(num_boxes):
                #得到相对于网格的左上角坐标
                # print('bxbybwbh[i][k].shape: {}'.format(bxbybwbh[i][k].shape))
                left_x ,top_y = int(bxbybwbh[i][k][2]),int(bxbybwbh[i][k][3])
                #得到当前网格的(left_x ,top_y)匹配最大iou的anchor索引号
                max_iou,max_index = torch.max(ious[i][top_y,left_x,:,k],dim = 0)
                j = max_index.item()
                cls_targets[i][j,labels[k],top_y,left_x] = 1
                #计算当前box的宽高对应iou最大的anchor宽高之间比值
                tw,th = bxbybwbh[i][k][2] / self.anchors[i][j][0],bxbybwbh[i][k][3] / self.anchors[i][j][1]
                #保存当前的宽高比以及左上角的left_x ,top_y
                # print('txty[i][k][0]: {}'.format(txty[i][k][0]))
                # print('txty[i][k][1]: {}'.format(txty[i][k][1]))
                loc_targets[i][j,:,top_y,left_x] = torch.Tensor([txty[i][k][0],txty[i][k][1],tw,th])
            t_boxes.append(boxes / gridSize[i])
        return loc_targets,cls_targets,t_boxes
    def decoder(self,output):
        """
        output:
            [1,(5 + num_classes)*3,13,13]
            [1,(5 + num_classes)*3,26,26]
            [1,(5 + num_classes)*3,52,52]
        """
        fmsize = []
        for i in range(len(self.S)):
            fmsize.append(self.S[i])
        #[1,125,13,13] => [5,25,13,13] == [5;(tx,ty,tw,th,c) + num_classes;13;13]
        outputs = []
        for i in range(len(self.S)):
            outputs.append(output[i].view(3,5 + self.num_classes,self.S[i],self.S[i]))

        #获得txty
        loc_txtys,loc_twths,confidences,probs,xys,box_xys = [],[],[],[],[],[]
        for i in range(len(outputs)):
            loc_txtys.append(outputs[i][:,:2,:,:])
            loc_twths.append(outputs[i][:,2:4,:,:])
            confidences.append(outputs[i][:,4,:,:])
            probs.append(outputs[i][:,5:,:,:])
            #生成网格 [fmsize,fmsize,2] => [2,fmsize,fmsize]
            xys.append(
                meshgrid_xy(fmsize[i]).view(fmsize[i],fmsize[i],2).permute(2,0,1))
            #[5,2,fmsize,fmsize]
            box_xys.append(loc_txtys[i].sigmoid() + xys[i].expand_as(loc_txtys[i]))

        # 转换anchor的类型为tensor
        box_whs = []
        for i in range(len(self.S)):
            anchors_wh = torch.tensor(self.anchors[i])
            anchors_wh = anchors_wh.view(self.B, 2,1,1).expand(self.B, 2, fmsize[i], fmsize[i])
            box_whs.append(loc_twths[i].exp() * anchors_wh)

        #根据中心坐标和高宽转换为[xmin,ymin,xmax,ymax] => [5,4,fmsize,fmsize]
        boxes,cls_preds, = [],[]
        for i in range(len(outputs)):
            box = torch.cat([box_xys[i] - box_whs[i] / 2,
                             box_xys[i] + box_whs[i] / 2], dim=1)
            box = box.permute(0,2,3,1).contiguous().view(-1,4)
            box /= fmsize[i]
            boxes.append(box)

            confidences[i] = confidences[i].sigmoid().view(-1)
            cls_preds.append(probs[i].permute(0,2,3,1).contiguous().view(-1,self.num_classes))
        for i in range(len(outputs)):
            cls_preds[i] = torch.softmax(cls_preds[i],dim=1)
        scores,cls_labels = [],[]
        for i in range(len(outputs)):
            score = cls_preds[i] * confidences[i].unsqueeze(dim = 1).expand_as(cls_preds[i])
            scores.append(score.max(dim=1)[0].view(-1))
            cls_labels.append(cls_preds[i].max(dim=1)[1].view(-1))
        return (torch.cat(boxes,dim = 0).view(-1,4),
                torch.cat(scores,dim = 0).view(-1),
               torch.cat(cls_labels,dim = 0).view(-1))


