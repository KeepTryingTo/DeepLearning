import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import bboxes_iou


class YOLOLayer(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """
    def __init__(
        self, config_model, layer_no, in_ch, ignore_thre=0.7):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
                GAUSSIAN (bool): predict uncertainty for each of xywh coordinates in Gaussian YOLOv3 way.
                    For Gaussian YOLOv3, see https://arxiv.org/abs/1904.04620
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(YOLOLayer, self).__init__()
        strides = [32, 16, 8] # fixed
        self.anchors = config_model['ANCHORS']
        #TODO 根据当前输出的第几层得到对应的anchor box索引
        self.anch_mask = config_model['ANCH_MASK'][layer_no]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = config_model['N_CLASSES']
        self.gaussian = config_model['GAUSSIAN']
        self.ignore_thre = ignore_thre
        self.stride = strides[layer_no]
        #TODO 根据下采样步长对anchor的高宽进行缩放
        all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        #TODO anchor mask其实就是表示使用anchor的索引（3个anchor box）
        self.masked_anchors = [all_anchors_grid[i]
                               for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)

        channels_per_anchor = 5 + self.n_classes  # 5: x, y, w, h, objectness
        if self.gaussian:
            print('Gaussian YOLOv3')
            channels_per_anchor += 4  # 4: xywh uncertainties
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=channels_per_anchor * self.n_anchors,
                              kernel_size=1, stride=1, padding=0)

    def forward(self, xin, labels=None):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            labels (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N and K denote batchsize and number of labels.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """
        output = self.conv(xin)

        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5 + self.n_classes  # channels per anchor w/o xywh unceartainties
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor
        #TODO [b,3,channels_per_anchor=num_cls + 5, fs,fs]
        output = output.view(batchsize, self.n_anchors, -1, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2)  # shape: [batch, anchor, grid_y, grid_x, channels_per_anchor]
        #TODO 对坐标采用高斯建模
        if self.gaussian:
            #TODO  logistic activation for sigma of xywh（output的最后四个元素）
            sigma_xywh = output[..., -4:]  # shape: [batch, anchor, grid_y, grid_x, 4(= xywh uncertainties)]
            sigma_xywh = torch.sigmoid(sigma_xywh)

            output = output[..., :-4]#TODO 从开始到倒数第四个元素之前的所有元素
        # output shape: [batch, anchor, grid_y, grid_x, n_class + 5(= x, y, w, h, objectness)]

        # TODO logistic activation for xy, obj, cls
        #TODO np.r_ 用于拼接数组或创建数组的快捷方式。它提供了一种简洁的方式来进行沿特定轴（通常是第一轴，即行轴）的数组拼接
        #TODO 将[x,y,obj_c,...num_cls]缩放0-1之间
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
            output[..., np.r_[:2, 4:n_ch]])

        # calculate pred - xywh obj cls
        #TODO 计算网格坐标
        x_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32), output.shape[:4]))
        y_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))

        masked_anchors = np.array(self.masked_anchors)

        w_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), shape = output.shape[:4]))
        h_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), shape = output.shape[:4]))

        #TODO 根据网格坐标以及anchor box的高宽得到预测结果
        pred = output.clone()
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

        #TODO 如果是测试阶段
        if labels is None:  # not training
            pred[..., :4] *= self.stride #TODO 根据步长将预测结果缩放至相对原图大小
            pred = pred.contiguous().view(batchsize, -1, n_ch)  # shsape: [batch, anchor x grid_y x grid_x, n_class + 5]
            #TODO
            if self.gaussian:
                # TODO 以求得的方差均值作为不确定性 scale objectness confidence with xywh uncertainties
                sigma_xywh = sigma_xywh.contiguous().view(batchsize, -1, 4)  # shsape: [batch, anchor x grid_y x grid_x, 4]
                sigma = sigma_xywh.mean(dim=-1) #TODO 求解[x,y,w,h]的均值
                pred[..., 4] *= (1.0 - sigma) #TODO 然后obj_c * （1 - uncertaintie）得到一定的score

                # TODO unnormalize uncertainties
                sigma_xywh = torch.sqrt(sigma_xywh)
                sigma_xywh[..., :2] *= self.stride
                sigma_xywh[..., 2:]  = torch.exp(sigma_xywh[..., 2:])

                #TODO concat pred with uncertainties
                # shsape: [batch, anchor x grid_y x grid_x, n_class + 9]
                pred = torch.cat([pred, sigma_xywh], 2)  #

            return pred.data

        pred = pred[..., :4].data  # shape: [batch, anchor, grid_y, grid_x, 4(= x, y, w, h)]

        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors,
                               fsize, fsize, 4 + self.n_classes).type(dtype)
        obj_mask = torch.ones(batchsize, self.n_anchors,
                              fsize, fsize).type(dtype)
        tgt_scale = torch.zeros(batchsize, self.n_anchors,
                                fsize, fsize, 2).type(dtype)

        target = torch.zeros(batchsize, self.n_anchors,
                             fsize, fsize, n_ch).type(dtype)

        #TODO 得到标签以及统计目标的数量
        labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = labels[:, :, 1] * fsize
        truth_y_all = labels[:, :, 2] * fsize
        truth_w_all = labels[:, :, 3] * fsize
        truth_h_all = labels[:, :, 4] * fsize
        truth_i_all = truth_x_all.to(torch.int16).numpy()
        truth_j_all = truth_y_all.to(torch.int16).numpy()

        #TODO 针对batch 遍历每一张图像
        for b in range(batchsize):
            #TODO 获得当前图像中的物体数
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = dtype(np.zeros((n, 4)))
            #TODO 获得当前图像所有box的高宽
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # TODO calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors)
            #TODO 计算和gt box的IOU最大的那些anchor box索引
            best_n_all = np.argmax(anchor_ious_all, axis=1)
            #TODO 得到真实anchor box对应的索引
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_mask[0]) | (
                best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]
            #TODO 计算预测的box和gt box之间的IOU
            pred_ious = bboxes_iou(pred[b].contiguous().view(-1, 4),
                                   truth_box, xyxy=False)
            #TODO 统计和预测box最大IOU的那些gt box
            pred_best_iou, _ = pred_ious.max(dim=1)
            #TODO 过滤掉那些低置信度的结果
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.contiguous().view(pred[b].shape[:3]) #[batch, anchor, grid_y, grid_x]
            #TODO  set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~pred_best_iou

            if sum(best_n_mask) == 0:
                continue

            for ti in range(best_n.shape[0]):
                #TODO 判断当前是否有匹配的anchor box
                if best_n_mask[ti] == 1:
                    #TODO 获得对应gt box的[x,y]
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    #TODO 将对应的位置设置为1
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1

                    #TODO 获得相对网格单元的坐标
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - \
                        truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - \
                        truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    #TODO gt box的高宽编码
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    #TODO 对应的标签位置设置为1
                    target[b, a, j, i, 5 + labels[b, ti,
                                                  0].to(torch.int16).numpy()] = 1
                    #TODO 高宽比率
                    tgt_scale[b, a, j, i, :] = 2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize

        # TODO loss calculation
        output[..., 4] *= obj_mask
        output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask

        #TODO 计算置信度和类别概率
        loss_obj = F.binary_cross_entropy(output[..., 4], target[..., 4], reduction='sum')
        loss_cls = F.binary_cross_entropy(output[..., 5:], target[..., 5:], reduction='sum')


        if self.gaussian:
            #TODO 负对数似然损失
            loss_xy = - torch.log(
                self._gaussian_dist_pdf(output[..., :2], target[..., :2], sigma_xywh[..., :2]) + 1e-9) / 2.0
            loss_wh = - torch.log(
                self._gaussian_dist_pdf(output[..., 2:4], target[..., 2:4], sigma_xywh[..., 2:4]) + 1e-9) / 2.0
        else:
            loss_xy = F.binary_cross_entropy(output[..., :2], target[..., :2], reduction='none')
            loss_wh = F.mse_loss(output[..., 2:4], target[..., 2:4], reduction='none') / 2.0
        loss_xy = (loss_xy * tgt_scale).sum()
        loss_wh = (loss_wh * tgt_scale).sum()

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls

    def _gaussian_dist_pdf(self, val, mean, var):
        return torch.exp(- (val - mean) ** 2.0 / var / 2.0) / torch.sqrt(2.0 * np.pi * var)
