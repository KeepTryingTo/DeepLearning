'''
# -*- encoding: utf-8 -*-
# 文件    : utils.py
# 说明    : 
# 时间    : 2022/06/28 16:19:01
# 作者    : Hito
# 版本    : 1.0
# 环境    : pytorch1.7
'''
import torch
import numpy as np



def adjust_LR(optimizer, epoch):
    lr = 1e-9
    if epoch < 5:
        lr = 1e-9
    elif epoch >= 5 and epoch < 10:
        lr = 2e-9
    elif epoch >= 10 and epoch < 15:
        lr = 4e-9
    else:
        lr = 1e-9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


    
def mask_by_sel(loss_mask,
                pos_indices,
                neg_indices):
    """
    cpu side calculation
    :param loss_mask:
    :param pos_indices: N×4dim
    :param neg_indices:
    :return:
    """

    assert loss_mask.size() == torch.Size([loss_mask.size(0), 1, 60, 60])

    # print('=> before fill loss mask:%d non_zeros.' % torch.nonzero(loss_mask).size(0))

    for pos_idx in pos_indices:
        loss_mask[pos_idx[0], pos_idx[1], pos_idx[2], pos_idx[3]] = 1.0

    for row in range(neg_indices.size(0)):
        for col in range(neg_indices.size(1)):
            idx = int(neg_indices[row][col])

            if idx < 0 or idx >= 3600:
                # print('=> idx: ', idx)
                continue

            y = idx // 60
            x = idx % 60

            try:
                loss_mask[row, 0, y, x] = 1.0
            except Exception as e:
                print(row, y, x)
                
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def NMS(dets,
        nms_thresh=0.4):
    """
    Pure Python NMS baseline
    :param dets:
    :param nms_thresh:
    :return:
    """
    #得到左上角坐标和右下角坐标
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]

    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)

        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= nms_thresh)[0]

        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return keep

def parse_out_MN(score_map,
                 loc_map,
                 M,
                 N,
                 K=10):
    """
    parse output from arbitrary input image size M×N
    M: image height, M rows
    N: image width, N cols
    """
    #由于经过网络之后的图像是经过了4倍的下采样之后的
    assert score_map.size() == torch.Size([1, 1, M // 4, N // 4])  # N×C×H×W
    assert loc_map.size() == torch.Size([1, 4, M // 4, N // 4])

    # squeeze output
    #score_map: [4, M // 4, N // 4]; loc_map: [M // 4, N // 4]
    score_map, loc_map = score_map.squeeze(), loc_map.squeeze()

    # reshape output, score_map: 1×(M×N), loc_map:4×(M×N)
    score_map, loc_map = score_map.view(1, -1), loc_map.view(4, -1)

    # filter out top k bbox with highest 
    score_map = torch.sigmoid(score_map)
    #由于之后车牌号一个类别，所以对于获取前K个最大类别分数以及索引时，K = 1
    #由于score_map: [1,N * M] => dim = 1
    scores, indices = torch.topk(input=score_map,
                                 k=K,
                                 dim=1)

    indices = [indices.squeeze()]
    score_map = score_map.squeeze().data

    dets = []
    cols_out = N // 4  # cols in output coordinate space
    for idx in indices:
        idx = int(idx)
        #得到有物体的像素点的x,y坐标
        xi, yi = idx % cols_out, idx // cols_out

        #当前的像素点坐标 - 距离左上角的距离和距离右下角的距离
        xt = xi - loc_map[0, idx] * (M // 4)
        yt = yi - loc_map[1, idx] * (N // 4)
        xb = xi - loc_map[2, idx] * (M // 4)
        yb = yi - loc_map[3, idx] * (N // 4)

        # map back to input coordinate space
        xt = float(xt.data) * 4.0
        yt = float(yt.data) * 4.0
        xb = float(xb.data) * 4.0
        yb = float(yb.data) * 4.0

        det = [xt, yt, xb, yb, float(score_map[idx])]

        dets.append(det)

    return np.array(dets)