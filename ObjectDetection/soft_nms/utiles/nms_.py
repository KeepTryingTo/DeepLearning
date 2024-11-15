"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/11/7-19:38
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import cv2

import torch
import torch.nn as nn
import numpy as np
# from torchvision.ops import nms

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    box_a = box_a.view(-1,4)
    box_b = box_b.view(-1,4)

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        xx1 = torch.index_select(x1, 0, idx)
        yy1 = torch.index_select(y1, 0, idx)
        xx2 = torch.index_select(x2, 0, idx)
        yy2 = torch.index_select(y2, 0, idx)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def two_box_IoU(box1, box2):
    # box = [x1, y1, x2, y2]
    zero = torch.tensor(0.0)
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    intersection = torch.max(zero, x2 - x1) * torch.max(zero, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def cal_weight_box(boxes,scores,max_conf_box,num_boxes):
    # TODO ================================================================================
    # TODO 保存计算的权重
    weights = torch.zeros(num_boxes).to(boxes.device)
    for i in range(num_boxes):
        weights[i] = scores[i] * two_box_IoU(boxes[i], max_conf_box)
    # TODO 权重和boxes相乘之后的结果
    weighted_box = torch.sum(weights.unsqueeze(dim=1) *
                             boxes, dim=0) / torch.sum(weights)
    # TODO ================================================================================
    return weighted_box

def nmw(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    v, idx = scores.sort(0)  # sort in ascending order

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    #TODO 选择前topk个框进行筛选
    idx = idx[-top_k:]  # indices of the top-k largest vals
    w = boxes.new()
    h = boxes.new()
    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        #TODO 把当前的最大置信度所对应的索引保留
        idx = idx[:-1]  # remove kept element from view
        # TODO load bboxes of next highest vals
        xx1 = torch.index_select(x1, 0, idx)
        yy1 = torch.index_select(y1, 0, idx)
        xx2 = torch.index_select(x2, 0, idx)
        yy2 = torch.index_select(y2, 0, idx)
        # TODO store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        #TODO 将张量 w 的形状调整为与张量 xx2 相同的形状
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        #TODO 这里我们选择置信度大于指定阈值的作为weighted box的计算
        #TODO 也就是计算那些和当前框IOU比较大，通过这些框进行调整置信度最大的那个框
        gt_iou_idx = idx[IoU.ge(overlap)]
        weighted_box = cal_weight_box(boxes[gt_iou_idx],
                                      scores[gt_iou_idx],
                                      max_conf_box=boxes[i],
                                      num_boxes=len(gt_iou_idx))

        # TODO 计算保留那些和最大置信度框IOU小于指定阈值的框
        idx = idx[IoU.le(overlap)]
        boxes[i] = weighted_box

    return keep, count, boxes

def soft_nms(boxes,scores,labels,
             conf_threshold=0.5,
             nms_threshold=0.4,
             gamma = 0.5,
             soft_nms_fn='linear'):
    keep = []
    #TODO 将box + score + label给拼接起来
    dets = torch.cat((boxes, scores.unsqueeze(dim=1),
                      labels.unsqueeze(dim = 1)), dim=1)
    #TODO 过滤掉那些置信度低于指定阈值的
    dets = dets[dets[:, 4] > conf_threshold]

    #TODO SOFT_NMS正式进入
    while len(dets) > 0:
        #TODO 首先将置信度乘以对应标签的值（这个计算方式和yolov1中的定义差不多）
        #TODO 然后选择置信度最大的索引
        _, idx = torch.max(dets[:, 4] * dets[:, 5], dim=0)
        #TODO 判断当前索引所对应的置信度是否大于指定的阈值
        val = dets[idx, 4]
        if val <= conf_threshold:
            continue
        pd = dets[idx]
        #TODO 将除了pd的box之外的其他box进行拼接
        dets = torch.cat((dets[:idx], dets[idx + 1:]))
        #TODO 计算置信度最大box和剩余box之间的IOU
        ious = jaccard(pd[:4], dets[:, :4])
        #TODO 将那些大于指定IOU并且和那些具有与最大置信度相同类别的进行“与”操作（就是只针对那些相同类别的框进行筛选）
        mask = (ious > nms_threshold) & (pd[-1] == dets[:, -1])
        mask = mask.squeeze()
        ious = ious.squeeze()
        #TODO 将当前的box加入候选集
        keep.append(pd)
        if mask.size() == torch.Size([]):
            break
        if dets.numel() == 1:
            break
        #TODO 对于那些置信度box高于指定阈值的进行soft计算
        if soft_nms_fn == 'linear':
            # print('ious.shape: {}'.format(ious.size()))
            # print('mask.shape: {}'.format(mask.size()))
            # print('dets.shape: {}'.format(dets.size()))
            dets[mask, 4] = dets[mask,4] * (1 - ious[mask]) * (1 - val)
        else:
            dets[mask, 4] = dets[mask,4] * torch.exp(-ious[mask] / gamma)
        #TODO 再来进一步赛选掉那些低置信度的框
        dets = dets[dets[:, 4] > conf_threshold]

    if len(keep) == 0:
        return torch.tensor(keep)
    return torch.stack(keep,dim = 0)

def demo():
    boxes = torch.tensor([
        [50, 50, 150, 150],
        [60, 60, 140, 140],
        [70, 70, 130, 130],
        [200, 200, 300, 300],
        [55, 65, 350, 350],
        [66, 66, 240, 140],
        [73, 77, 230, 230],
        [100, 150, 350, 350]
    ])
    confidences = torch.tensor([0.7, 0.66, 0.65, 0.6,
                                0.23,0.45,0.55,
                                0.33])
    labels = torch.tensor([1] * len(boxes))
    dets = soft_nms(boxes,confidences,labels)

    print('dets.size: {}'.format(dets.size(0)))
    for det in dets:
        print('det: {}'.format(det))

    img = np.zeros((400, 400, 3), dtype=np.uint8)
    for det in dets:
        box = det[:4]
        cv2.rectangle(img, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (255, 0, 0), 2)

    cv2.imshow('Image with Bounding Boxes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo()
    pass