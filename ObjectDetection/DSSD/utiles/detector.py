"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/2/7 18:39
"""

"""
score_threshold=0.3
nms_iou_threshold=0.2
max_detection_boxes_num=150
"""
import torch
from torch import nn
from models.fcos import FCOS
from collections import OrderedDict
from configs.config import DefaultConfig
from utiles.encoder import coords_fmap2orig


class DetectHead(nn.Module):
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides, config=None):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num
        # strides=[8,16,32,64,128]
        self.strides = strides
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config

    def forward(self, inputs):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        cls_logits, coords = self._reshape_cat_out(inputs[0], self.strides)  # [batch_size,sum(_h*_w),class_num]
        cnt_logits, _ = self._reshape_cat_out(inputs[1], self.strides)  # [batch_size,sum(_h*_w),1]
        reg_preds, _ = self._reshape_cat_out(inputs[2], self.strides)  # [batch_size,sum(_h*_w),4]

        # 计算预测classes和center_ness的sigmoid
        cls_preds = cls_logits.sigmoid_()
        cnt_preds = cnt_logits.sigmoid_()

        # max: return max value and max value's index得到对应的预测概率和对应的最大概率索引
        cls_scores, cls_classes = torch.max(cls_preds, dim=-1)  # [batch_size,sum(_h*_w)]
        if self.config.add_centerness:
            # 计算预测类别概率 X 预测得到的center_ness
            cls_scores = cls_scores * (cnt_preds.squeeze(dim=-1))  # [batch_size,sum(_h*_w)]
        cls_classes = cls_classes + 1  # [batch_size,sum(_h*_w)]

        # 将预测得到的reg_preds偏移加上coords网格坐标 [l*,t*,r*,b*] => [xmin,ymin,xmax,ymax]
        boxes = self._coords2boxes(coords, reg_preds)  # [batch_size,sum(_h*_w),4]

        # select topk 选择前K个预测结果
        max_num = min(self.max_detection_boxes_num, cls_scores.shape[-1])
        # 得到前K个最大scores值的索引;largest表示返回第k个最小值
        topk_ind = torch.topk(cls_scores, max_num, dim=-1, largest=True, sorted=True)[1]  # [batch_size,max_num]
        _cls_scores = []
        _cls_classes = []
        _boxes = []
        # 针对batchsize进行遍历
        for batch in range(cls_scores.shape[0]):
            # 得到相应的前K中的类别概率
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])  # [max_num]
            # 得到相应的前K中的类别索引
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])  # [max_num]
            # 得到相应的前K中的物体预测框
            _boxes.append(boxes[batch][topk_ind[batch]])  # [max_num,4]
        cls_scores_topk = torch.stack(_cls_scores, dim=0)  # [batch_size,max_num]
        cls_classes_topk = torch.stack(_cls_classes, dim=0)  # [batch_size,max_num]
        boxes_topk = torch.stack(_boxes, dim=0)  # [batch_size,max_num,4]
        assert boxes_topk.shape[-1] == 4
        return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _post_process(self, preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post = []
        _cls_classes_post = []
        _boxes_post = []
        cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
        # 根据前面选择的前K个最高类别scores选择大于给定阈值的类别scoreS，类别索引以及预测类别的边框
        for batch in range(cls_classes_topk.shape[0]):
            # 得到当前图像中物体的scores大于给定阈值的scores，mask中包含了相应索引位置为False或者True
            mask = cls_scores_topk[batch] >= self.score_threshold #根据score阈值分数进行初筛
            _cls_scores_b = cls_scores_topk[batch][mask]  # [?]
            _cls_classes_b = cls_classes_topk[batch][mask]  # [?]
            _boxes_b = boxes_topk[batch][mask]  # [?,4]

            # 通过NMS，获得大于给定阈值的边框索引 通过NMS算法进一步筛选重叠的框
            nms_ind = self.batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores, classes, boxes = torch.stack(_cls_scores_post, dim=0), torch.stack(_cls_classes_post,
                                                                                   dim=0), torch.stack(_boxes_post,
                                                                                                       dim=0)

        return scores, classes, boxes

    @staticmethod
    def box_nms(boxes, scores, thr):
        '''
        NMS的基本流程
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # 根据预测类别scores进行降序排序，order中保存的是排序之后相应的索引
        order = scores.sort(0, descending=True)[1]
        # 保存最后留下来的边框
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                # 得到当前最大scores的索引
                i = order[0].item()
                keep.append(i)

            xmin = x1[order[1:]].clamp(min=float(x1[i]))
            ymin = y1[order[1:]].clamp(min=float(y1[i]))
            xmax = x2[order[1:]].clamp(max=float(x2[i]))
            ymax = y2[order[1:]].clamp(max=float(y2[i]))
            inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            idx = (iou <= thr).nonzero().squeeze()
            if idx.numel() == 0:
                break
            # 对于当前的IOU小于给定的阈值IOU，那么进行下一轮的操作
            order = order[idx + 1]
        return torch.LongTensor(keep)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):
        """
        boxes = _boxes_b
        scores = _cls_scores_b
        idxs = _cls_classes_b
        iou_threshold = self.nms_iou_threshold
        """
        # 如果不存在给定的边框，那么返回0
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        """
        策略:为了每个类独立地执行NMS。
        我们给所有的盒子添加一个偏移量。偏移量是相关的
        只对类idx，并且是足够大的盒子
        来自不同类别的边框不重叠
        """
        #得到最大坐标值
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self, coords, offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        xmin = x - l*, ymin = y - t*
        xmax = x + r*, ymax = y + b*
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size,sum(_h*_w),4]
        return boxes

    def _reshape_cat_out(self, inputs, strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size = inputs[0].shape[0]
        c = inputs[0].shape[1]
        out = []
        coords = []
        for pred, stride in zip(inputs, strides):
            # [batch_size,c,_h,_w] => [batch_size,_h,_w,c]
            pred = pred.permute(0, 2, 3, 1)
            # coord: [256,2]
            coord = coords_fmap2orig(pred, stride).to(device=pred.device)
            # pred: [batch_size,_h x _w,c]
            pred = torch.reshape(pred, [batch_size, -1, c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out, dim=1), torch.cat(coords, dim=0)


class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_imgs, batch_boxes):
        batch_boxes = batch_boxes.clamp_(min=0)
        h, w = batch_imgs.shape[2:]
        batch_boxes[..., [0, 2]] = batch_boxes[..., [0, 2]].clamp_(max=w - 1)
        batch_boxes[..., [1, 3]] = batch_boxes[..., [1, 3]].clamp_(max=h - 1)
        return batch_boxes


class FCOSDetector(nn.Module):
    def __init__(self, weight_path = r'' ,config=None):
        super().__init__()
        if config is None:
            config = DefaultConfig
        self.fcos_body = FCOS(config=config)
        state_dict = OrderedDict()
        checkpoint = torch.load(weight_path,map_location='cpu')
        for key in list(checkpoint.keys()):
            value = checkpoint[key]
            state_dict[key[10:]] = value
        self.fcos_body.load_state_dict(state_dict)
        """
        score_threshold=0.3
        nms_iou_threshold=0.2
        max_detection_boxes_num=150
        """
        self.detection_head = DetectHead(config.score_threshold, config.nms_iou_threshold,
                                         config.max_detection_boxes_num, config.strides, config)
        self.clip_boxes = ClipBoxes()

    def forward(self, inputs):
        """
        inputs
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        """
        batch_imgs = inputs
        out = self.fcos_body(batch_imgs)
        scores, classes, boxes = self.detection_head(out)
        boxes = self.clip_boxes(batch_imgs, boxes)
        return scores, classes, boxes
