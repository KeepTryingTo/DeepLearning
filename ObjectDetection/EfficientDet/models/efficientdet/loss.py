import torch
import torch.nn as nn
import cv2
import numpy as np

from models.efficientdet.utils import BBoxTransform, ClipBoxes
from utiles.utils import postprocess, invert_affine, display

#a:[49104,,4]; b: [num_boxes,4]
def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    #计算gt box的area: [num_boxes]; iw,ih: [49104,num_boxes]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    #[49104,num_boxes]
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih
    IoU = intersection / ua
    #IOU: [49104,num_boxes]
    return IoU

"""
    feature map.shape: torch.Size([1, 64, 64, 64])
    regression.shape: torch.Size([1, 64, 32, 32])
    classification.shape: torch.Size([1, 64, 16, 16])
    anchor.shape: torch.Size([1, 64, 8, 8])
"""
class FocalLoss(nn.Module):
    def __init__(self,device):
        super(FocalLoss, self).__init__()
        self.device = device
    #classifications: [batch,49104,90]; regressions: [batch,49104,4]; anchors: [1,49104,4]; annotations: [batch,max_num_annots,4]
    def forward(self, classifications, regressions,
                anchors, annotations, **kwargs):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        #[49104,4]
        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        #[xmin,ymin,xmax,ymax] => [x,y,w,h];anchor_widths: [49104]
        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):
            #[49104,90]; [49104,4]
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            #得到第j张图像中的所有标注的boxes: [max_num_annotas,5]
            bbox_annotation = annotations[j]
            #针对背景进行过滤: [num_boxes,5]（根据数据打包的情况来看，只需要得到不为-1的值即为当前图像中实际的目标box）
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            #对预测得到的classification进行过滤一遍
            classification = torch.clamp(
                input=classification,
                min = 1e-4,
                max = 1.0 - 1e-4
            )

            #如果没有box的话，那么对classification使用focal loss计算损失值
            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = alpha_factor.to(self.device)
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
                    bce = -(torch.log(1.0 - classification))
                    
                    cls_loss = focal_weight * bce
                    
                    regression_losses.append(torch.tensor(0).to(dtype).to(self.device))
                    classification_losses.append(cls_loss.sum())
                else:
                    
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
                    bce = -(torch.log(1.0 - classification))
                    
                    cls_loss = focal_weight * bce
                    
                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(cls_loss.sum())

                continue

            #计算当前的anchors和对应的gt box的IOU
            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

            #计算最大的IOU值以及对应的索引,在行的维度(anchor维度)上找到和anchor匹配IOU最大的gt box索引以及最大IOU值；[49104]；[49104]
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification，定义一个和classification大小相同的矩阵，全是-1的值；[49104,90]
            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets.to(self.device)
            #将IOU值小于0.4的赋值为0
            targets[torch.lt(IoU_max, 0.4), :] = 0
            #将IOU值大于0.5的作为正样本 ，并且大于0.5的索引为True，否则为False；[49104]
            positive_indices = torch.ge(IoU_max, 0.5)
            #得到正样本的数量:int
            num_positive_anchors = positive_indices.sum()
            #得到分配给gt box的anchors： [49104,5]，得到为49104个anchor分配的gt box
            assigned_annotations = bbox_annotation[IoU_argmax, :]

            #将正样本的位置赋值为0
            targets[positive_indices, :] = 0
            targets[
                positive_indices,
                assigned_annotations[positive_indices, 4].long()
            ] = 1 #将gt分配给anchor的正样本位置label_id设置为1
            #[49104,90]
            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.to(self.device)
            #alpha_factor = (label == 1 : p else 1 - p): [49104,90]
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            #Pt = (label == 1 : p else 1 - p): [49104,90]
            focal_weight = torch.where(torch.eq(targets, 1.),
                                       1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma) #[49104,90]
            #[49104,90];gamma = 2.0
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            #[49104,90]
            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.to(self.device)
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros) #torch.ne()——判断元素是否不相等(判断target中不为-1的索引)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :] #[num_positives,5]
                #anchor_widths_pi,anchor_heights_pi,anchor_ctr_x_pi,anchor_ctr_y_pi => [num_postives] 得到gt box对应anchors box信息
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                #gt_widths，gt_heights，gt_ctr_x，gt_ctr_y => [num_postives]
                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style 对gt box中的最小值进行约束
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)
                # 计算gt box和anchor box之间的offset(进行归一化)
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)
                # [4,num_postives]
                targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t() #[num_postives,4]
                #计算target和predict_box之间的loss
                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).to(self.device))
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        # debug
        imgs = kwargs.get('imgs', None)
        if imgs is not None:
            #根据预测的predict_box和anchors box转换为预测得到的box[xmin,ymin,xmax,ymax]
            regressBoxes = BBoxTransform()
            #对预测得到的boxes进行边界处理
            clipBoxes = ClipBoxes()
            obj_list = kwargs.get('obj_list', None)
            out = postprocess(
                x = imgs.detach(),
                anchors = torch.stack([anchors[0]] * imgs.shape[0], 0).detach(),
                regression = regressions.detach(),
                classification = classifications.detach(),
                regressBoxes = regressBoxes,
                clipBoxes = clipBoxes,
                threshold = 0.5, iou_threshold = 0.3
            )
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
            display(out, imgs, obj_list, imshow=False, imwrite=True)

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True) * 50  # https://github.com/google/automl/blob/6fdd1de778408625c1faf368a327fe36ecd41bf7/efficientdet/hparams_config.py#L233
