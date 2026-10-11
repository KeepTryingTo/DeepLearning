import torch
from losses.commons import ct_focal_loss,giou_loss


class CenterNetLoss(object):
    def __init__(self,
                 hm_weight=1.,
                 wh_weight=5.,
                 down_ratio=4
                 ):
        self.hm_weight=hm_weight
        self.wh_weight=wh_weight
        self.down_ratio=down_ratio
        self.base_loc=None

    def __call__(self, pred_hm,
                      pred_wh,
                      heatmap,
                      box_target,
                      wh_weight):
        """

        Args:
            pred_hm: tensor, (batch, 80, h, w).
            pred_wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            heatmap: tensor, same as pred_hm.
            box_target: tensor, same as pred_wh.
            wh_weight: tensor, same as pred_wh.

        Returns:
           hm_loss
           wh_loss
        """
        ##----------------------------------------
        if pred_hm.dtype == torch.float16:
            pred_hm = pred_hm.float()
        if pred_wh.dtype == torch.float16:
            pred_wh = pred_wh.float()
        cur_device = pred_wh.device
        #TODO ground truth
        heatmap=heatmap.to(cur_device)
        box_target=box_target.to(cur_device)
        wh_weight=wh_weight.to(cur_device)

        ##----------------------------------------
        #TODO 获得输出特征图的高宽，并利用focusloss计算分类损失
        H, W = pred_hm.shape[2:]
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        hm_loss = ct_focal_loss(pred_hm, heatmap) * self.hm_weight  # classification loss

        mask = wh_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4

        #TODO 生成网格坐标
        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
            base_step = self.down_ratio  # 4
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        # TODO (batch, h, w, 4)
        pred_boxes = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
                                self.base_loc + pred_wh[:, [2, 3]]),
                               dim=1).permute(0, 2, 3, 1)
        # TODO (batch, h, w, 4)
        boxes = box_target.permute(0, 2, 3, 1)
        wh_loss = giou_loss(pred_boxes, boxes, mask,
                            avg_factor=avg_factor) * self.wh_weight

        loss = hm_loss+wh_loss
        return loss, torch.stack([hm_loss, wh_loss]).detach()


