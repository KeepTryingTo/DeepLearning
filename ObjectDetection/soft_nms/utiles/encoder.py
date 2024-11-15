"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/2/6 21:25
"""

import torch
import numpy as np
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,anchors,num_classes,overlap_threshold = 0.5):
        super(Encoder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.overlap_threshold = overlap_threshold
    def iou(self, box):
        #   计算出每个真实框与所有的先验框的iou;判断真实框与先验框的重合情况
        inter_upleft = np.maximum(self.anchors[:, :2], box[:2])
        inter_botright = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        area_true = (box[2] - box[0]) * (box[3] - box[1])

        area_gt = (self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])
        union = area_true + area_gt - inter
        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True, variances=[0.1, 0.1, 0.2, 0.2]):
        #   计算当前真实框和先验框的重合情况;iou [self.num_anchors];encoded_box [self.num_anchors, 5
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))
        #   找到每一个真实框，重合程度较高的先验框;真实框可以由这个先验框来负责预测
        assign_mask = iou > self.overlap_threshold
        #   如果没有一个先验框重合度大于self.overlap_threshold;则选择重合度最大的为正样本
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True#
        #   利用iou进行赋值
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        #   找到对应的先验框
        assigned_anchors = self.anchors[assign_mask]
        #   逆向编码，将真实框转化为ssd预测结果的格式;先计算真实框的中心与长宽
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        #   再计算重合度较高的先验框的中心与长宽
        assigned_anchors_center = (assigned_anchors[:, 0:2] + assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh = (assigned_anchors[:, 2:4] - assigned_anchors[:, 0:2])
        #   逆向求取ssd应该有的预测结果;先求取中心的预测结果，再求取宽高的预测结果;存在改变数量级的参数，默认为[0.1,0.1,0.2,0.2]
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center #真实框和anchor box之间的中心点距离
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]
        #展平操作
        return encoded_box.ravel()

    def assign_boxes(self, boxes):#
        #   assignment分为3个部分
        #   :4      的内容为网络应该有的回归预测结果
        #   4:-1    的内容为先验框所对应的种类，默认为背景
        #   -1      的内容为当前先验框是否包含目标
        # => [num_anchors,4(c_x,c_y,x_w,x_h) + background + (VOC: 0 - 20) + (0不包含物体 or 1包含物体)]
        assignment = np.zeros((self.num_anchors, 4 + self.num_classes + 1))
        assignment[:, 4] = 1.0 #confidence
        if len(boxes) == 0:
            return assignment
        # 对每一个真实框都进行iou计算 apply_along_axis:https://blog.csdn.net/starmoth/article/details/83832458
        #其中axis = 1表示作用在boxes的行上面
        encoded_boxes = np.apply_along_axis(self.encode_box, axis=1, arr=boxes[:, :4])
        #   在reshape后，获得的encoded_boxes的shape为：[num_true_box, num_anchors, 4 + 1];num_true_box表示gt box；4是编码后的结果，1为iou
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)
        #   [num_anchors]求取每一个先验框重合度最大的真实框，注意axis = 0表示作用在列上面
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        #   计算一共有多少先验框满足需求
        assign_num = len(best_iou_idx)
        # 将编码后的真实框取出
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        #   编码后的真实框的赋值
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        #   4代表为背景的概率，设定为0，因为这些先验框有对应的物体
        assignment[:, 4][best_iou_mask] = 0
        #赋值类别标签
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]
        #   -1表示先验框是否有对应的物体
        assignment[:, -1][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment