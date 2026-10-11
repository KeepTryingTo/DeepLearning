import torch

######################### Modified Focal Loss ----------------------------------------------


def ct_focal_loss(pred, gt, gamma=2.0):
    """
    Focal loss used in CornerNet & CenterNet. Note that the values in gt (label) are in [0, 1] since
    gaussian is used to reduce the punishment and we treat [0, 1) as neg example.

    Args:
        pred: tensor, any shape.
        gt: tensor, same as pred.
        gamma: gamma in focal loss.

    Returns:

    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)  # reduce punishment
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return neg_loss
    return (pos_loss + neg_loss) / num_pos





################## L1 loss -----------------------------------------------------------------


def giou_loss(pred,
              target,
              weight,
              avg_factor=None):
    """GIoU loss.
    Computing the GIoU loss between a set of predicted bboxes and target bboxes.
    """
    pos_mask = weight > 0
    weight = weight[pos_mask].float()
    if avg_factor is None:
        avg_factor = torch.sum(pos_mask).float().item() + 1e-6
    bboxes1 = pred[pos_mask].view(-1, 4)
    bboxes2 = target[pos_mask].view(-1, 4)

    #TODO 得到[left,top,right,bottom]
    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    #TODO 得到预测的box和gt box之间的重叠区域高宽
    wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
    #TODO 得到最小外接矩形的[xmin,ymin,xmax,ymax]
    enclose_x1y1 = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    enclose_x2y2 = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    #TODO 最小外接矩形的高宽
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)

    overlap = wh[:, 0] * wh[:, 1]
    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (ap + ag - overlap)

    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]  # i.e. C in paper
    u = ap + ag - overlap
    gious = ious - (enclose_area - u) / enclose_area
    iou_distances = 1 - gious
    return torch.sum(iou_distances * weight) / avg_factor
