"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/11/8-16:44
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import matplotlib.pyplot as plt
import math
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import json
import cv2
from tqdm import tqdm

voc_classes = {'__background__': 0, 'aeroplane': 1, 'bicycle': 2,
               'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7,
               'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12,
               'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16,
               'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}
voc_indices = dict([(voc_classes[k] - 1, k) for k in voc_classes])


def draw_bboxes(img, bboxes, color, th=1):
    img_ = img.copy()
    for bbox in bboxes:
        x, y, w, h = bbox
        pt1 = (int(x - w / 2), int(y - h / 2))
        pt2 = (int(x + w / 2), int(y + h / 2))
        img_ = cv2.rectangle(img_, pt1, pt2, color, thickness=th)
    return img_


def tensor_to_img(src):
    dst = np.transpose(src.cpu().numpy(), [1, 2, 0])
    return dst


class Logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.files = {'val': open(os.path.join(log_dir, 'val.txt'), 'a+'),
                      'train': open(os.path.join(log_dir, 'train.txt'), 'a+')}

    def write_line2file(self, mode, string):
        self.files[mode].write(string + '\n')
        self.files[mode].flush()

    def write_loss(self, epoch, losses, lr):
        tmp = str(epoch) + '\t' + str(lr) + '\t'
        print('Epoch', ':', epoch, '-', lr)
        writer = SummaryWriter(log_dir=self.log_dir)
        writer.add_scalar('lr', math.log(lr), epoch)
        for k in losses:
            if losses[k] > 0:
                writer.add_scalar('Train/' + k, losses[k], epoch)
                print(k, ':', losses[k])
                # self.writer.flush()
        tmp += str(round(losses['all'], 5)) + '\t'
        self.write_line2file('train', tmp)
        writer.close()

    def write_metrics(self, epoch, metrics, save=[], mode='Val', log=True):
        tmp = str(epoch) + '\t'
        print("validation epoch:", epoch)
        writer = SummaryWriter(log_dir=self.log_dir)
        for k in metrics:
            if k in save:
                tmp += str(metrics[k]) + '\t'
            if log:
                tag = mode + '/' + k
                writer.add_scalar(tag, metrics[k], epoch)
                # self.writer.flush()
            print(k, ':', metrics[k])

        self.write_line2file('val', tmp)
        writer.close()


def iou_wt_center(bbox1, bbox2):
    # only for torch, return a vector nx1
    bbox1 = bbox1.view(-1, 4)
    bbox2 = bbox2.view(-1, 4)

    # TODO tranfer xc,yc,w,h to xmin ymin xmax ymax
    xmin1 = bbox1[:, 0] - bbox1[:, 2] / 2
    xmin2 = bbox2[:, 0] - bbox2[:, 2] / 2

    ymin1 = bbox1[:, 1] - bbox1[:, 3] / 2
    ymin2 = bbox2[:, 1] - bbox2[:, 3] / 2

    xmax1 = bbox1[:, 0] + bbox1[:, 2] / 2
    xmax2 = bbox2[:, 0] + bbox2[:, 2] / 2

    ymax1 = bbox1[:, 1] + bbox1[:, 3] / 2
    ymax2 = bbox2[:, 1] + bbox2[:, 3] / 2

    inter_xmin = torch.max(xmin1, xmin2)
    inter_xmax = torch.min(xmax1, xmax2)
    inter_ymin = torch.max(ymin1, ymin2)
    inter_ymax = torch.min(ymax1, ymax2)

    #TODO 计算交集的高宽，并过滤掉那些不满足的交集box得到mask
    inter_w = inter_xmax - inter_xmin
    inter_h = inter_ymax - inter_ymin
    mask = ((inter_w >= 0) & (inter_h >= 0)).to(torch.float)

    # detect not overlap

    # inter_h[inter_h<0] = 0
    #TODO 计算交集的面积
    inter = inter_w * inter_h * mask
    # keep iou<0 to avoid gradient diasppear
    area1 = bbox1[:, 2] * bbox1[:, 3]
    area2 = bbox2[:, 2] * bbox2[:, 3]
    union = area1 + area2 - inter
    ious = inter / union
    #TODO
    ious[ious != ious] = torch.tensor(0.0, device=bbox1.device)
    return ious


def iou_wt_center_np(bbox1, bbox2):
    # in numpy,only for evaluation,return a matrix m x n
    bbox1 = bbox1.reshape(-1, 4)
    bbox2 = bbox2.reshape(-1, 4)

    # tranfer xc,yc,w,h to xmin ymin xmax ymax
    xmin1 = bbox1[:, 0] - bbox1[:, 2] / 2
    xmin2 = bbox2[:, 0] - bbox2[:, 2] / 2
    ymin1 = bbox1[:, 1] - bbox1[:, 3] / 2
    ymin2 = bbox2[:, 1] - bbox2[:, 3] / 2
    xmax1 = bbox1[:, 0] + bbox1[:, 2] / 2
    xmax2 = bbox2[:, 0] + bbox2[:, 2] / 2
    ymax1 = bbox1[:, 1] + bbox1[:, 3] / 2
    ymax2 = bbox2[:, 1] + bbox2[:, 3] / 2

    # trigger broadcasting
    inter_xmin = np.maximum(xmin1.reshape(-1, 1), xmin2.reshape(1, -1))
    inter_xmax = np.minimum(xmax1.reshape(-1, 1), xmax2.reshape(1, -1))
    inter_ymin = np.maximum(ymin1.reshape(-1, 1), ymin2.reshape(1, -1))
    inter_ymax = np.minimum(ymax1.reshape(-1, 1), ymax2.reshape(1, -1))

    inter_w = inter_xmax - inter_xmin
    inter_h = inter_ymax - inter_ymin
    mask = ((inter_w >= 0) & (inter_h >= 0))

    # inter_h[inter_h<0] = 0
    inter = inter_w * inter_h * mask.astype(float)
    area1 = ((ymax1 - ymin1) * (xmax1 - xmin1)).reshape(-1, 1)
    area2 = ((ymax2 - ymin2) * (xmax2 - xmin2)).reshape(1, -1)
    union = area1 + area2 - inter
    ious = inter / union
    ious[ious != ious] = 0
    return ious


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            if plot:
                plt.plot(recall_curve, precision_curve)
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def cal_tp_per_item(pds, gts, threshold=0.5):
    assert (len(pds.shape) > 1) and (len(gts.shape) > 1)
    pds = pds.cpu().numpy()
    gts = gts.cpu().numpy()
    n = pds.shape[0]
    tps = np.zeros(n)
    labels = np.unique(gts[:, 0].astype(np.int))
    scores = pds[:, 4] * pds[:, 5]
    idx = np.argsort(-scores)
    scores = scores[idx]
    pds = pds[idx]
    ##print(len(labels))
    for c in labels:
        pd_idx = np.where(pds[:-1] == c)[0]
        pdbboxes = pds[pd_idx, :4].reshape(-1, 4)
        gtbboxes = gts[gts[:, 0] == c, 1:].reshape(-1, 4)
        ##print(voc_indices[int(c)])
        ##print(pdbboxes)
        ##print(gtbboxes)
        nc = pdbboxes.shape[0]
        mc = gtbboxes.shape[0]
        selected = np.zeros(mc)
        sel_ious = np.zeros(mc)
        for i in range(nc):
            if mc == 0:
                break
            pdbbox = pdbboxes[i]
            ious = iou_wt_center_np(pdbbox, gtbboxes)
            iou = ious.max()
            best = ious.argmax()
            ##print(iou)
            if iou >= threshold and selected[best] != 1:
                selected[best] = 1
                tps[pd_idx[i]] = 1.0
                mc -= 1
                sel_ious[best] = iou
    return [tps, scores, pds[:, -1]]


def cal_tp_per_item_wo_cls(pds, gts, threshold=0.5):
    assert (len(pds.shape) > 1) and (len(gts.shape) > 1)
    pds = pds.cpu().numpy()
    gts = gts.cpu().numpy()
    ##print(gts.shape,pds.shape)
    n = pds.shape[0]
    tps = np.zeros(n)
    scores = pds[:, 4] * pds[:, 5]
    idx = np.argsort(-scores)
    scores = scores[idx]
    pds = pds[idx]
    pd_idx = np.where(pds[:-1] < 21)[0]
    pdbboxes = pds[pd_idx, :4].reshape(-1, 4)
    gtbboxes = gts[:, 1:].reshape(-1, 4)
    nc = pdbboxes.shape[0]
    mc = gtbboxes.shape[0]
    selected = np.zeros(mc)
    for i in range(nc):
        if mc == 0:
            break
        pdbbox = pdbboxes[i]
        ious = iou_wt_center_np(pdbbox, gtbboxes)
        best = ious.argmax()
        iou = ious.max()
        if (selected[best] != 1) and (iou > threshold):
            selected[best] = 1
            tps[pd_idx[i]] = 1.0
            mc -= 1
    return [tps, scores, pds[:, -1]]


def xyhw2xy(boxes_):
    boxes = boxes_.clone()
    boxes[:, 0] = boxes_[:, 0] - boxes_[:, 2] / 2
    boxes[:, 1] = boxes_[:, 1] - boxes_[:, 3] / 2
    boxes[:, 2] = boxes_[:, 0] + boxes_[:, 2] / 2
    boxes[:, 3] = boxes_[:, 1] + boxes_[:, 3] / 2
    return boxes


def xy2xyhw(boxes):
    boxes_ = boxes.clone()
    boxes_[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    boxes_[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    boxes_[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes_[:, 3] = boxes[:, 3] - boxes[:, 1]
    return boxes_


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    return boxes


def cal_metrics_wo_cls(pd, gt, threshold=0.5):
    pd = pd.cpu().numpy()  # n
    gt = gt.cpu().numpy()  # m
    pd_bboxes = pd[:, :4]
    gt = gt[:, 1:]
    m = len(gt)
    n = len(pd_bboxes)
    if n > 0 and m > 0:
        ious = iou_wt_center_np(pd_bboxes, gt)  # nxm
        scores = ious.max(axis=1)
        fp = scores <= threshold

        # only keep trues
        ious = ious[~fp, :]
        fp = fp.sum()  # transfer to scalar

        select_ids = ious.argmax(axis=1)
        # discard fps hit gt boxes has been hitted by bboxes with higher conf
        tp = len(np.unique(select_ids))
        fp += len(select_ids) - tp

        # groud truth with no associated predicted object
        assert (fp + tp) == n
        fn = m - tp
        p = tp / n
        r = tp / m
        assert (p <= 1)
        assert (r <= 1)
        ap = tp / (fp + fn + tp)
        return p, r, ap
    elif m > 0 or n > 0:
        return 0, 0, 0
    else:
        return 1, 1, 1


def non_maximum_supression(preds, conf_threshold=0.5, nms_threshold=0.4):
    preds = preds[preds[:, 4] > conf_threshold]
    if len(preds) == 0:
        return preds
    score = preds[:, 4] * preds[:, 5:].max(1)[0]
    idx = torch.argsort(score, descending=True)
    preds = preds[idx]
    cls_confs, cls_labels = torch.max(preds[:, 5:], dim=1, keepdim=True)
    dets = torch.cat((preds[:, :5], cls_confs, cls_labels.float()), dim=1)
    keep = []
    while len(dets) > 0:
        mask = dets[0, -1] == dets[:, -1]
        new = dets[0]
        keep.append(new)
        ious = iou_wt_center(dets[0, :4], dets[:, :4])
        if not (ious[0] >= 0.7):
            ious[0] = 1
        mask = mask & (ious > nms_threshold)
        # hard-nms
        dets = dets[~mask]
    if len(keep) > 0:
        return torch.stack(keep).reshape(-1, 7)
    else:
        return torch.tensor(keep).reshape(-1, 7)

#TODO https://github.com/Pamikk/obj_det_loss/blob/master/utils.py
def non_maximum_supression_soft(preds, conf_threshold=0.5, nms_threshold=0.4):
    keep = []
    #TODO 根据输出的类别分数，然后在类别分数维度上求得最大预测类别分数和对应的类别
    cls_confs, cls_labels = torch.max(preds[:, 5:], dim=1, keepdim=True)
    #TODO 将box + score + label给拼接起来
    dets = torch.cat((preds[:, :5], cls_confs, cls_labels.float()), dim=1)
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
        ious = iou_wt_center(pd[:4], dets[:, :4])
        #TODO 将那些大于指定IOU并且和那些具有与最大置信度相同类别的进行“与”操作（就是只针对那些相同类别的框进行筛选）
        mask = (ious > nms_threshold) & (pd[-1] == dets[:, -1])
        #TODO 将当前的box加入候选集
        keep.append(pd)
        #TODO 对于那些置信度box高于指定阈值的进行soft计算
        dets[mask, 4] *= (1 - ious[mask]) * (1 - val)
        #TODO 再来进一步赛选掉那些低置信度的框
        dets = dets[dets[:, 4] > conf_threshold]
    if len(keep) > 0:
        return torch.stack(keep).reshape(-1, 7)
    else:
        return torch.tensor(keep).reshape(-1, 7)