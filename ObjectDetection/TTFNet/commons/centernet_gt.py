import numpy as np
import torch
from commons.boxs_utils import bbox_areas




class CenterNetGT(object):
    def __init__(self,
                 alpha=0.54,
                 beta=0.54,
                 num_cls=80,
                 wh_planes=4,
                 down_ratio=4,
                 wh_area_process='log'
                 ):
        self.alpha=alpha
        self.beta=beta
        self.num_cls = num_cls
        self.wh_planes = wh_planes
        self.wh_area_process = wh_area_process
        self.down_ratio=down_ratio

    #TODO 根据给定的形状和σ生成指定的高斯分布
    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h


    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        #TODO 生成高斯分布的高度和宽度
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]
        #TODO 得到ground truth中高斯分布的区域
        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                          w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def target_single_image(self, gt_boxes, gt_labels, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple. (h,w) featuremap size

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w)
            reg_weight: tensor, tensor <=> img, (1, h, w)
        """
        output_h, output_w = feat_shape
        heatmap_channel = self.num_cls  # 80

        #TODO 保存生成的gt heatmap，gt box以及回归权重
        heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w))
        box_target = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        reg_weight = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))

        #TODO 框面积的处理方式
        if self.wh_area_process == 'log':
            boxes_areas_log = bbox_areas(gt_boxes).log()  # [num_gt,]
        elif self.wh_area_process == 'sqrt':
            boxes_areas_log = bbox_areas(gt_boxes).sqrt()
        else:
            boxes_areas_log = bbox_areas(gt_boxes)

        #TODO 按照面积大小获取其值
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log,
                                                    boxes_areas_log.size(0))

        #TODO 是否需要进行归一化处理
        if self.wh_area_process == 'norm':
            boxes_area_topk_log[:] = 1.

        # TODO 对gtbox按size进行排列
        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        #TODO 根据下采样比率，也需要对box进行缩放
        feat_gt_boxes = gt_boxes / self.down_ratio
        #TODO 检测box的边界情况
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]],
                                               min=0, max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]],
                                               min=0, max=output_h - 1)
        #TODO 获得box的高宽
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])  # shape=[gt_num,] feature尺度

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        # TODO 获得box中心坐标 shape=[gt_num,2]  feature尺度
        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)
        #TODO 对box的高度和宽度进行一定的缩放
        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        #TODO 再一次对box的高宽进行缩放
        if self.alpha != self.beta:
            h_radiuses_beta = (feat_hs / 2. * self.beta).int()
            w_radiuses_beta = (feat_ws / 2. * self.beta).int()

        # TODO 遍历box larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            #TODO 获得box对应的类别标签
            cls_id = gt_labels[k].int().item()   # why -1 ???

            fake_heatmap = fake_heatmap.zero_()
            #TODO 通过高斯函数获得对应box区域的heatmap
            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(),
                                        w_radiuses_alpha[k].item())

            #TODO 当两个中心坐标点靠的很近的时候，生成的heatmap必然存在重叠，那么获得两个中最大的那个值
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)  # 优先预测较小的gt box

            #TODO 根据缩放之后的box高宽来得到heatmap
            if self.alpha != self.beta:
                fake_heatmap = fake_heatmap.zero_()
                self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                            h_radiuses_beta[k].item(),
                                            w_radiuses_beta[k].item())
            box_target_inds = fake_heatmap > 0

            box_target[:, box_target_inds] = gt_boxes[k][:, None]
            cls_id = 0

            #TODO 和论文的公式一一对应
            local_heatmap = fake_heatmap[box_target_inds]
            ct_div = local_heatmap.sum()
            local_heatmap *= boxes_area_topk_log[k]
            reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div

        return heatmap, box_target, reg_weight

    def __call__(self, targets, img_shape, bs):
        """
        Args:
            targets(tensor): shape=[-1,7]   7==>(batch_id,weight,label_idx,x1,y1,x2,y2)
            img_shape (list): [h,w]  input tensor size
            bs

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w)
            reg_weight: tensor, (batch, 1, h, w)
        """
        with torch.no_grad():
            #TODO 获得网络输出特征图的高宽
            feat_shape = (img_shape[0] // self.down_ratio,
                          img_shape[1] // self.down_ratio)
            #TODO 保存生成的gt heatmap，gt box以及回归权重
            heatmaps, box_targets, reg_weights=list(),list(),list()

            for bi in range(bs):
                #TODO 获得当前batch中索引对应的图像target
                target = targets[targets[:, 0] == bi, 2:]  # shape=[num_gt,5] 5==>(label_idx,x1,y1,x2,y2)
                gt_boxes, gt_labels = target[:, 1:], target[:, [0]]

                heatmap, box_target, reg_weight = self.target_single_image(
                                                    gt_boxes,
                                                    gt_labels,
                                                    feat_shape=feat_shape)
                heatmaps.append(heatmap)
                box_targets.append(box_target)
                reg_weights.append(reg_weight)

            heatmaps, box_targets = [torch.stack(t, dim=0).detach() for t in [heatmaps, box_targets]]
            reg_weights = torch.stack(reg_weights, dim=0).detach()

            return heatmaps, box_targets, reg_weights

