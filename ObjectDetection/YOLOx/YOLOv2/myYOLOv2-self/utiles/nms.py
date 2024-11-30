
import time
import numpy as np
import torch
import torchvision
from configs.config import *
from utiles.encoder import Encoder

"""
.cpu()如果你想把tensor从GPU传回到CPU
.detach()梯度信息去掉
"""
def to_cpu(tensor):
    return tensor.detach().cpu()

def cxcywhToxyxy(boxes):
    """
    :param boxes: [[cx,cy,w,h],...] : [num_boxes,4]
    :return:
    """
    box_xy = torch.FloatTensor(torch.zeros(size=(4,)))
    #[cx,cy,w,h] => [xmin,ymin,xmax,ymax]
    box_xy[0] = boxes[0] - 0.5 * boxes[2]
    box_xy[1] = boxes[1] - 0.5 * boxes[3]
    box_xy[2] = boxes[0] + 0.5 * boxes[2]
    box_xy[3] = boxes[1] + 0.5 * boxes[3]
    return box_xy

def convert_cellboxes(
        predictions,conf_threshold = 0.1,
):
    """
    :param predictions: 预测得到的结果[B,125,13,13]
    :return:
    """
    batch_size = predictions.shape[0]
    encoder = Encoder(
            anchors=ANCHORS,img_size=IMG_SIZE,
            S = S,B = B,num_classes=VOC_NUM_CLASSES
        )
    boxes,scores,labels = encoder.decoder(predictions)
    mask = (scores > conf_threshold).nonzero(as_tuple=False).squeeze()
    boxes,scores,labels = boxes[mask],scores[mask],labels[mask]
    is_exist_object = True
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        scores = torch.zeros(1)
        labels = torch.zeros(1)
        is_exist_object = False
    else:
        boxes = boxes.view(-1,4)  #(4,) => (1,4)
        scores = scores.view(-1)
        labels = labels.view(-1)
    ###############################################################################
    return boxes,scores,labels,is_exist_object

#计算non_max_suppression的方式一
def nms(
        boxes,scores,labels,
        iou_threshold = 0.1
):
    """
    NMS主要是去掉冗余的框，比如重叠的框
    :param boxes: 经过初筛之后的box
    :param probs:
    :param cls_indexes:
    :return:
    """
    # 获取预测框左上角和右下角坐标
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # 计算预测框面积
    areas = (x2 - x1) * (y2 - y1)
    # 降序排序
    t_scores = scores
    _, order = t_scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        # print('numel: ',order.numel())
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0]
            keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= iou_threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    keep = torch.LongTensor(keep)
    #经过NMS之后的boxes，scores以及class index
    r_boxes = boxes[keep]
    r_scores = scores[keep]
    class_label = labels[keep]
    return r_boxes,r_scores,class_label

def non_max_suppression(outputs, conf_thres=0.9, iou_thres=0.45, classes=2):
    """
    :param outputs: 网络输出的结果 [b,7,7,num_classes + B * 5]
    :param conf_thres: 给定的置信度阈值
    :param iou_thres: 给定的IOU阈值
    :param classes: 类别数
    :return:
    """
    boxes = torch.cat(
        [
            outputs[...,classes:classes + 4].unsqueeze(0),
            outputs[...,classes + 5:classes + 9].unsqueeze(0)
        ],dim = 0
    ).view(-1,4)
    scores = torch.cat([
        outputs[...,classes + 4].unsqueeze(0),
        outputs[...,classes + 9].unsqueeze(0)
    ],dim = 0).view(-1,1)
    labels = outputs[...,:classes].view(-1,1)
    #scores输出的结果是按从大到小的顺序排列的，所以根据conf_thres = 0.7排除小于此值的所有框
    scores = torch.unsqueeze(scores,dim = 1)
    labels = torch.unsqueeze(labels,dim = 1)
    # print('scores.shape: {}'.format(scores.shape))
    # print('labels.shape: {}'.format(labels.shape))
    #将其进行拼接
    detections = torch.cat((boxes,scores,labels),dim = 1)
    #由于这里检测的是一张图像，所以在第一个维度进行升维
    detections = torch.unsqueeze(detections,dim = 0)
    # print('detection.shape: {}'.format(detections.shape))

    #图像中包含的最多的检测框的数量max_nms
    max_nms = 300
    # maximum number of detections per image
    max_det = 300
    #限制检测时间
    time_limit = 1.0
    t = time.time()

    # output保存输出的[x1,y1,w,h,conf,cls]
    # prediction.shape[0]表示batch_size
    output = [torch.zeros((0, 6), device="cpu")] * detections.shape[0]

    for xi, x in enumerate(detections):
        """
        detections.shape = [candiates_box_num,6] => [(xmin,ymin,xmax,ymax,confidence,class_id)]
        对于边框高宽超过给定阈值的直接赋值为0
        detections[((detections[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        输出的x中包含置信度大于阈值的保留
        """
        # print('x[0]_0: {}'.format(x[0]))
        #首先过滤掉小于给定置信度阈值的框
        x = x[x[..., 4] > conf_thres]  # confidence
        # If none remain process next image
        #x.shape = [,85]
        # print('x.shape: {}'.format(x.shape))
        #如果一张图像中的候选框已经筛选完毕，那么执行下一张图像中物体的候选框筛选
        if not x.shape[0]:
            continue
        # sort by confidence
        # 如果候选框的数量大于给定的阈值框数量，那么按从从大小的顺序
        # 排列，并且只取前max_nms的框
        n = x.shape[0]
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        # print('x.shape: {}'.format(x.shape))
        boxes,scores,lables = x[:,:4],x[:,4],x[:,5]
        """
               前面首先是通过置信度筛选出候选框
               输出的i值为int64张量与已保留的元素的索引按NMS，按分数降序排序
               """
        # print('boxes.shape: {}'.format(boxes.shape))
        # boxes = torch.unsqueeze(boxes, dim = 1)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        # 限制每一张图像中候选框的最大数量，也就是一张图像中最多可以检测max_det个物体
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        """
        将x[i]对应经过NMS的框加载到CPU上并且去掉梯度信息
        """
        output[xi] = to_cpu(x[i])

        # 在规定的时间内进行NMS
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    #boxes: output[:,:4],scores: output[:,4],label: output[:,5]
    return output

def rescale_boxes(boxes, current_dim, original_shape):
    """
    :param boxes: 框的坐标
    :param current_dim: 当前输入到网络的图像尺寸大小
    :param original_shape: 当前检测的图像大小
    :return:
    """
    #获得原始图像的高宽
    orig_h, orig_w = original_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed
    """
    如果宽度比长度大的话，那么对高度进行填充
    如果长度比宽度大的话，那么对宽度进行填充
    填充的目的就是让长和宽的大小一致
    """
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image
    #根据图像的缩放对其坐标和高宽进行缩放
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

#计算non_max_suppression方式二
# def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
#     """
#     Does Non Max Suppression given bboxes
#
#     Parameters:
#         bboxes (list): list of lists containing all bboxes with each bboxes
#         specified as [class_pred, prob_score, x1, y1, x2, y2]
#         iou_threshold (float): threshold where predicted bboxes is correct
#         threshold (float): threshold to remove predicted bboxes (independent of IoU)
#         box_format (str): "midpoint" or "corners" used to specify bboxes
#
#     Returns:
#         list: bboxes after performing NMS given a specific IoU threshold
#     """
#
#     assert type(bboxes) == list
#
#     bboxes = [box for box in bboxes if box[1] > threshold]
#     bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
#     bboxes_after_nms = []
#
#     while bboxes:
#         chosen_box = bboxes.pop(0)
#
#         bboxes = [
#             box
#             for box in bboxes
#             if box[0] != chosen_box[0]
#                or intersection_over_union(
#                 torch.tensor(chosen_box[2:]),
#                 torch.tensor(box[2:])
#             )
#                < iou_threshold
#         ]
#
#         bboxes_after_nms.append(chosen_box)
#
#     return bboxes_after_nms