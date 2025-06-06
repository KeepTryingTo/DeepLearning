"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/6 13:17
"""

import torch
from torch import nn
from torchvision.models import resnet50
import torch.nn.functional as F
from .matcher import build_matcher
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        #实例化的匹配box算法
        self.matcher = matcher
        """
        weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
        weight_dict['loss_giou'] = args.giou_loss_coef
        """
        self.weight_dict = weight_dict
        #无对象类的相对分类权值
        self.eos_coef = eos_coef
        #losses = ['labels','boxes','cardinality']
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        #设置背景类别的权重为eos_coef
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        计算类别标签损失值
        """
        assert 'pred_logits' in outputs
        #得到预测的类别
        src_logits = outputs['pred_logits']

        #batch_idx,src_idx
        idx = self._get_src_permutation_idx(indices)
        #其中J表示predict的box在target中的最佳匹配索引
        #target_classes_o表示预测的box在target中最佳匹配索引box在target中对应的类别标签 == 根据最佳匹配，预测的box在target对应的label
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        #建立一个全是背景类别的tensor
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)

        #将预测的结果对应在target中的类别赋值
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        计算基数误差，即预测的非空框数量的绝对误差。这不是真正的损失，它仅用于记录目的。它不传播梯度
        """
        #得到预测的类别
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        #转换为tensor
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # 将target的box在prediction中box匹配索引获取
        idx = self._get_src_permutation_idx(indices)
        #根据target的box匹配的结果，得到prediction中的对应预测box
        src_boxes = outputs['pred_boxes'][idx]
        #根据prediction的box在target中匹配的索引得到target中box的信息
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        #计算giou损失值
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        #batch_idx表示indices中匹配的num_boxes框属于batch中的哪张图像
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """
        outputs: 对应pred_logit预测类别以及pred_box预测框
        targets: 对应box以及labels
        indices: 最大匹配之后predict的box索引和target的box对应索引
        num_boxes:表示target中object的数量
        """
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        #outputs: pred_logits: [BS,num_queries,num_classes] and pred_boxes: [BS,num_queries,4]
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        #计算predict的box和target的box之间的最大匹配
        #返回值为：预测的box索引对应target的box索引
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        #统计一个batch中的所有object数
        num_boxes = sum(len(t["labels"]) for t in targets)
        #转换为tensor
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        #如果分布式训练可用
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        #根据分布式的数量得到num_boxes
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        #self.losses = {'labels','boxes','cardinality'}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes)
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        #对于计算辅助损失值，首先需要计算辅助分支的box和target的box最大匹配
        #然后其损失计算过程和上面的一样
        if 'aux_outputs' in outputs:
             for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        #得到预测的类别以及box
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2 #original image size [h,w]

        #计算类别的softmax概率值
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1) #得到概率值以及对应的类别

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1) #得到图像的高宽
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        #根据原始图像大小，将box还原回原图比例的box
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DETR(nn.Module):
    def __init__(
            self,num_classes,hidden_dim=256,nhead=8,
            num_encoder_layers=6,num_decoder_layers=6
    ):
        super(DETR, self).__init__()
        self.backbone = nn.Sequential(
            *list(resnet50(pretrained = True).children())[:-2]
        )
        self.conv = nn.Conv2d(in_channels=2048,out_channels=hidden_dim,kernel_size=(1,1))
        self.transformer = nn.Transformer(
            d_model=hidden_dim,nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers
        )

        self.linear_class = nn.Linear(in_features=hidden_dim,out_features=num_classes + 1)
        self.linear_bbox = nn.Linear(in_features=hidden_dim,out_features=4)
        self.query_pos = nn.Parameter(torch.rand(size = (100,hidden_dim)))
        self.row_embed = nn.Parameter(torch.rand(size = (50,hidden_dim // 2)))
        self.col_embed = nn.Parameter(torch.rand(size=(50, hidden_dim // 2)))

    def forward(self,input:NestedTensor):
        samples = input
        if isinstance(input, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(input)
        x = self.backbone(samples.tensors)
        h = self.conv(x)
        H,W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H,1,1),
            self.row_embed[:H].unsqueeze(1).repeat(1,W,1)
        ],dim = -1).flatten(0,1).unsqueeze(1)
        h = self.transformer(
            pos + h.flatten(2).permute(2,0,1),
            self.query_pos.unsqueeze(1)
        ).transpose(0, 1)
        cls = self.linear_class(h)
        bbox = self.linear_bbox(h).sigmoid()
        out = {'pred_logits': cls, 'pred_boxes': bbox}
        return out

def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    #全景分割
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    #根据构建的backbone和transformer最后得到dert模型
    model = DETR(
        num_classes=num_classes,hidden_dim=256,nhead=8,
        num_encoder_layers=6,num_decoder_layers=6
    )
    #对于图像分割的时候使用
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    #构建匈牙利匹配的实例
    matcher = build_matcher(args)
    #对应的损失权重系数
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    #在分割过程中使用
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    #TODO this is a hack 是否使用辅助分支
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    #如果进行分割的话
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    #如果进行图像分割的话的后处理
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors