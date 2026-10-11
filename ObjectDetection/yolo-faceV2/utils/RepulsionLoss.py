import torch
import numpy as np
from utils.general import box_iou

# reference: https://github.com/dongdonghy/repulsion_loss_pytorch/blob/master/repulsion_loss.py
def IoG(gt_box, pre_box):
    inter_xmin = torch.max(gt_box[:, 0], pre_box[:, 0])
    inter_ymin = torch.max(gt_box[:, 1], pre_box[:, 1])
    inter_xmax = torch.min(gt_box[:, 2], pre_box[:, 2])
    inter_ymax = torch.min(gt_box[:, 3], pre_box[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    #TODO 计算得到的交集面积和真实box面积的比值
    G = ((gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])).clamp(1e-6)
    return I / G

def smooth_ln(x, deta=0.5):
    return torch.where(
        torch.le(x, deta),
        -torch.log(1 - x),
        ((x - deta) / (1 - deta)) - np.log(1 - deta)
    )

# YU 添加了detach，减小了梯度对gpu的占用
def repulsion_loss_torch(pbox, gtbox, deta=0.5, pnms=0.1, gtnms=0.1, x1x2y1y2=False):
    repgt_loss = 0.0
    repbox_loss = 0.0
    pbox = pbox.detach()
    gtbox = gtbox.detach()
    gtbox_cpu = gtbox.cuda().data.cpu().numpy()
    #TODO 计算预测框和gt box之间的IOU
    pgiou = box_iou(pbox, gtbox, x1y1x2y2=x1x2y1y2)
    pgiou = pgiou.cuda().data.cpu().numpy()
    #TODO 计算预测的box之间IOU
    ppiou = box_iou(pbox, pbox, x1y1x2y2=x1x2y1y2)
    ppiou = ppiou.cuda().data.cpu().numpy()
    # t1 = time.time()
    len = pgiou.shape[0]
    #TODO 遍历每一个组
    for j in range(len):
        for z in range(j, len):
            ppiou[j, z] = 0
            #TODO 对于当前的组，判断第j个box和第z个box之间的坐标关系
            if ((gtbox_cpu[j][0]==gtbox_cpu[z][0]) and
                    (gtbox_cpu[j][1]==gtbox_cpu[z][1]) and
                    (gtbox_cpu[j][2]==gtbox_cpu[z][2]) and
                    (gtbox_cpu[j][3]==gtbox_cpu[z][3])):
                #TODO 如果存在重叠就设将其IOU设置为0
                pgiou[j, z] = 0
                pgiou[z, j] = 0
                ppiou[z, j] = 0

    # t2 = time.time()
    # print("for cycle cost time is: ", t2 - t1, "s")
    pgiou = torch.from_numpy(pgiou).cuda().detach()
    ppiou = torch.from_numpy(ppiou).cuda().detach()
    # TODO repgt
    max_iou, argmax_iou = torch.max(pgiou, 1)
    pg_mask = torch.gt(max_iou, gtnms) #TODO 判断过滤之后的IOU是否大于指定的NMS阈值
    num_repgt = pg_mask.sum()
    if num_repgt > 0:
        #TODO 获得预测框和真实框之间IOU
        iou_pos = pgiou[pg_mask, :]
        #TODO 计算最大IOU
        max_iou_sec, argmax_iou_sec = torch.max(iou_pos, 1)
        pbox_sec = pbox[pg_mask, :]
        gtbox_sec = gtbox[argmax_iou_sec, :]
        #TODO 和论文中给定的公式是一一对应的
        IOG = IoG(gtbox_sec, pbox_sec)
        repgt_loss = smooth_ln(IOG, deta)
        repgt_loss = repgt_loss.mean()

    # repbox
    pp_mask = torch.gt(ppiou, pnms)  # 防止nms为0, 因为如果为0,那么上面的for循环就没有意义了 [N x N] error
    num_pbox = pp_mask.sum()
    if num_pbox > 0:
        repbox_loss = smooth_ln(ppiou, deta)
        repbox_loss = repbox_loss.mean()
    # mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
    # print(mem)
    torch.cuda.empty_cache()

    return repgt_loss, repbox_loss