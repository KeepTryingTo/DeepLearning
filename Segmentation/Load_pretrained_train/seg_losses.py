"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2026/10/11
@Desc   : 分割损失函数库：CE / Dice / Focal / CE+Dice / Focal+Dice，
          统一处理 torchvision 分割模型的 dict 输出（out + aux 辅助头）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IGNORE_INDEX, LOSS_OPTIONS


def ce_loss(logits, target, ignore_index=IGNORE_INDEX):
    """交叉熵（忽略 255 像素；全部被忽略时返回 0 避免出现 nan）"""
    if (target != ignore_index).sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def dice_loss(logits, target, num_classes, ignore_index=IGNORE_INDEX, eps=1e-6):
    """多类软 Dice Loss。logits [N,C,H,W]，target [N,H,W]（类别 id）"""
    valid = (target != ignore_index)
    tgt = torch.where(valid, target, torch.zeros_like(target))
    target_onehot = F.one_hot(tgt, num_classes).permute(0, 3, 1, 2).float()
    # 忽略像素在 one-hot 与概率中同时置零，不参与交并集
    valid_f = valid.unsqueeze(1).float()
    target_onehot = target_onehot * valid_f
    prob = F.softmax(logits, dim=1) * valid_f

    dims = (0, 2, 3)
    inter = (prob * target_onehot).sum(dim=dims)
    card = prob.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice = (2.0 * inter + eps) / (card + eps)
    # 只对真实出现（GT 中存在）的类求平均
    present = (target_onehot.sum(dim=dims) > 0).float()
    return 1.0 - (dice * present).sum() / present.sum().clamp(min=1.0)


def focal_loss(logits, target, num_classes, gamma=2.0, alpha=None,
               ignore_index=IGNORE_INDEX):
    """多类 Focal Loss。

    :param gamma: 调制指数，关注难分样本
    :param alpha: 类别权重；None=均匀，float=前景加权，list/tensor=逐类权重（长度=num_classes）
    """
    valid = (target != ignore_index)
    if valid.sum() == 0:
        return logits.sum() * 0.0
    tgt = torch.where(valid, target, torch.zeros_like(target))
    logp = F.log_softmax(logits, dim=1)
    logp_t = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()
    loss = -((1.0 - p_t) ** gamma) * logp_t
    if alpha is not None:
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=logits.dtype, device=logits.device)
        loss = loss * alpha[tgt]
    return loss[valid].mean()


class SegCriterion(nn.Module):
    """统一损失入口。

    forward(outputs, target) 中：
    - outputs 为 dict（torchvision 模型训练时返回 {'out':…, 'aux':…}）：
      loss = L(out) + aux_weight * L(aux)
    - outputs 为 Tensor：loss = L(out)
    """

    def __init__(self, num_classes, loss_name="ce", ignore_index=IGNORE_INDEX,
                 aux_weight=0.5, focal_gamma=2.0, focal_alpha=None):
        super().__init__()
        if loss_name not in LOSS_OPTIONS:
            raise ValueError(f"未知损失: {loss_name}，可选: {LOSS_OPTIONS}")
        self.num_classes = num_classes
        self.loss_name = loss_name
        self.ignore_index = ignore_index
        self.aux_weight = float(aux_weight)
        self.focal_gamma = float(focal_gamma)
        if focal_alpha is not None and not isinstance(focal_alpha, (list, tuple)):
            focal_alpha = [float(focal_alpha)] * num_classes
        self.focal_alpha = focal_alpha

    def _one(self, logits, target):
        name = self.loss_name
        if name == "ce":
            return ce_loss(logits, target, self.ignore_index)
        if name == "dice":
            return dice_loss(logits, target, self.num_classes, self.ignore_index)
        if name == "focal":
            return focal_loss(logits, target, self.num_classes, self.focal_gamma,
                              self.focal_alpha, self.ignore_index)
        if name == "ce_dice":
            return (ce_loss(logits, target, self.ignore_index)
                    + dice_loss(logits, target, self.num_classes, self.ignore_index))
        if name == "focal_dice":
            return (focal_loss(logits, target, self.num_classes, self.focal_gamma,
                               self.focal_alpha, self.ignore_index)
                    + dice_loss(logits, target, self.num_classes, self.ignore_index))
        raise ValueError(f"未知损失: {name}")

    def forward(self, outputs, target):
        if isinstance(outputs, dict):
            loss = self._one(outputs["out"], target)
            aux = outputs.get("aux")
            if aux is not None and self.aux_weight > 0:
                loss = loss + self.aux_weight * self._one(aux, target)
            return loss
        return self._one(outputs, target)


if __name__ == "__main__":
    # 自测：各损失对 dict/Tensor 输出均可反传
    torch.manual_seed(0)
    N, C, H, W = 2, 4, 16, 16
    logits = torch.randn(N, C, H, W, requires_grad=True)
    aux = torch.randn(N, C, H, W, requires_grad=True)
    target = torch.randint(0, C, (N, H, W))
    target[0, :4] = IGNORE_INDEX
    for name in LOSS_OPTIONS:
        crit = SegCriterion(C, loss_name=name, aux_weight=0.5)
        l1 = crit({"out": logits, "aux": aux}, target)
        l2 = crit(logits, target)
        l1.backward(retain_graph=True)
        assert torch.isfinite(l1) and torch.isfinite(l2)
        print(f"{name:12s} dict={l1.item():.4f} tensor={l2.item():.4f}")
    print("seg_losses self-test passed")
