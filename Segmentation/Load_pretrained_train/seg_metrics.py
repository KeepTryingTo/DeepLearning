"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2026/10/11
@Desc   : 分割评估指标：基于混淆矩阵的完整指标套件
          Pixel Acc / Mean Acc / Mean Precision / mIoU / FWIoU / Mean Dice / Kappa
          以及逐类 IoU、Dice、Precision、Recall。
"""

import numpy as np

from config import IGNORE_INDEX


class SegMetrics:
    """累积混淆矩阵并计算全套分割指标。

    用法：
        meter = SegMetrics(num_classes)
        meter.update(pred_np, target_np)   # 多次累积，形状任意但需一致
        overall, per_class = meter.compute()
    """

    def __init__(self, num_classes, ignore_index=IGNORE_INDEX):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target):
        """pred/target: numpy 数组或 torch 张量（形状一致），值域 [0, C-1] 或 ignore_index"""
        if not isinstance(pred, np.ndarray):
            pred = pred.detach().cpu().numpy()
        if not isinstance(target, np.ndarray):
            target = target.detach().cpu().numpy()
        pred = pred.astype(np.int64).reshape(-1)
        target = target.astype(np.int64).reshape(-1)
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        pred = pred[valid]
        target = target[valid]
        if target.size == 0:
            return
        # 行=真实类别，列=预测类别
        idx = target * self.num_classes + pred
        cm = np.bincount(idx, minlength=self.num_classes ** 2)
        self.cm += cm.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """返回 (overall dict, per_class list[dict])"""
        cm = self.cm.astype(np.float64)
        total = cm.sum()
        diag = np.diag(cm)
        row = cm.sum(axis=1)   # 真实各类像素数
        col = cm.sum(axis=0)   # 预测各类像素数

        with np.errstate(divide="ignore", invalid="ignore"):
            precision = diag / col           # 预测为 c 的像素中正确的比例
            recall = diag / row              # 真实为 c 的像素中被召回的比例
            iou = diag / (row + col - diag)
            dice = 2.0 * diag / (row + col)

        present = (row + col) > 0            # 真实或预测中出现过的类
        gt_present = row > 0                 # 真实出现过的类

        overall = {
            "pixel_acc": float(diag.sum() / total) if total > 0 else 0.0,
            "mean_acc": float(np.nanmean(np.where(gt_present, recall, np.nan)))
                        if gt_present.any() else 0.0,
            "mean_precision": float(np.nanmean(np.where(present, precision, np.nan)))
                              if present.any() else 0.0,
            "miou": float(np.nanmean(np.where(present, iou, np.nan)))
                    if present.any() else 0.0,
            "fwiou": float((row / total * np.nan_to_num(iou)).sum())
                     if total > 0 else 0.0,
            "mean_dice": float(np.nanmean(np.where(present, dice, np.nan)))
                         if present.any() else 0.0,
        }
        # Kappa 一致性系数
        if total > 0:
            po = diag.sum() / total
            pe = float((row * col).sum()) / (total * total)
            overall["kappa"] = float((po - pe) / (1.0 - pe)) if pe < 1.0 else 1.0
        else:
            overall["kappa"] = 0.0

        per_class = []
        for c in range(self.num_classes):
            per_class.append({
                "iou": float(iou[c]) if present[c] else float("nan"),
                "dice": float(dice[c]) if present[c] else float("nan"),
                "precision": float(precision[c]) if present[c] else float("nan"),
                "recall": float(recall[c]) if gt_present[c] else float("nan"),
                "support": int(row[c]),
            })
        return overall, per_class

    def summary_str(self, classes=None):
        """逐类指标表格（控制台打印用）"""
        if classes is None:
            classes = [f"class_{i}" for i in range(self.num_classes)]
        overall, per_class = self.compute()
        lines = [f"{'class':<16s}{'IoU':>8s}{'Dice':>8s}{'P':>8s}{'R':>8s}{'pixels':>10s}"]
        for c, m in enumerate(per_class):
            name = classes[c] if c < len(classes) else f"class_{c}"
            def fmt(v):
                return f"{v:>8.4f}" if np.isfinite(v) else f"{'--':>8s}"
            lines.append(f"{name:<16s}{fmt(m['iou'])}{fmt(m['dice'])}"
                         f"{fmt(m['precision'])}{fmt(m['recall'])}{m['support']:>10d}")
        lines.append("-" * 58)
        lines.append(f"mIoU={overall['miou']:.4f} | mDice={overall['mean_dice']:.4f} | "
                     f"pAcc={overall['pixel_acc']:.4f} | mAcc={overall['mean_acc']:.4f} | "
                     f"FWIoU={overall['fwiou']:.4f} | Kappa={overall['kappa']:.4f}")
        return "\n".join(lines)


def torch_no_grad_if_available(func):
    """（保留给外部复用）以 no_grad 上下文执行被装饰函数"""
    import functools
    import torch

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper


if __name__ == "__main__":
    # 自测：完美预测 + 已知错误
    meter = SegMetrics(num_classes=3)
    t = np.array([0, 0, 1, 1, 2, 2])
    meter.update(t.copy(), t.copy())           # 完美
    meter.update(np.array([0, 1, 1, 1, 2, 0]), t)  # 2 个错
    overall, per_class = meter.compute()
    # 手算核对：两次 update 后 diag=[3,4,3]，总 12，col=[4,5,3]
    assert abs(overall["pixel_acc"] - 10 / 12) < 1e-9
    assert abs(overall["miou"] - np.mean([3 / 5, 4 / 5, 3 / 4])) < 1e-9
    print(meter.summary_str(["bg", "cat", "dog"]))
    print("seg_metrics self-test passed")
