"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2026/10/11
@Desc   : 可视化工具：mask 上色、叠加、原图/标签/预测对比图、训练曲线绘制、数据集预览网格。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示环境（服务器/脚本）下保存图片而非弹窗
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from config import IMAGENET_MEAN, IMAGENET_STD, palette_for

_RESAMPLE = getattr(Image, "Resampling", Image)


def denormalize(img_t):
    """[3,H,W] 归一化张量 -> [H,W,3] uint8"""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_t.cpu() * std + mean
    img = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def _mask_n_classes(mask):
    """mask 中出现过的类别数（忽略 255）"""
    m = np.asarray(mask)
    valid = m[m != 255]
    return int(valid.max()) + 1 if valid.size else 1


def colorize(mask, palette=None):
    """类别 id mask -> RGB 彩色图 [H,W,3]；忽略索引(255)渲染为白色"""
    mask = np.asarray(mask).astype(np.int64)
    valid = mask != 255
    if palette is None:
        palette = palette_for(max(_mask_n_classes(mask), 2))
    palette = np.asarray(palette, dtype=np.uint8)
    out = np.zeros(mask.shape + (3,), dtype=np.uint8)
    if valid.any():
        out[valid] = palette[np.clip(mask[valid], 0, len(palette) - 1)]
    out[~valid] = [255, 255, 255]
    return out


def overlay(img_rgb, color_mask, alpha=0.5):
    """原图与彩色 mask 半透明叠加"""
    img_rgb = np.asarray(img_rgb, dtype=np.float32)
    color_mask = np.asarray(color_mask, dtype=np.float32)
    if img_rgb.shape[:2] != color_mask.shape[:2]:
        color_mask = np.array(Image.fromarray(color_mask.astype(np.uint8)).resize(
            (img_rgb.shape[1], img_rgb.shape[0]), _RESAMPLE.NEAREST), dtype=np.float32)
    return (img_rgb * (1 - alpha) + color_mask * alpha).astype(np.uint8)


def save_comparison(orig, gt, pred, path, palette=None, alpha=0.5):
    """保存一行四联对比图：原图 | 标签 | 预测 | 叠加效果

    :param orig: [H,W,3] uint8 或 [3,H,W] 归一化张量
    :param gt / pred: [H,W] 类别 id 数组/张量
    """
    if isinstance(orig, torch.Tensor):
        orig = denormalize(orig)
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if palette is None:
        palette = palette_for(max(_mask_n_classes(gt), _mask_n_classes(pred), 2))

    gt_color = colorize(gt, palette)
    pred_color = colorize(pred, palette)
    pred_overlay = overlay(orig, pred_color, alpha)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (im, title) in zip(axes, [
            (orig, "Image"), (gt_color, "GroundTruth"),
            (pred_color, "Prediction"), (pred_overlay, "Overlay")]):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_history(history, save_path):
    """根据训练历史（list[dict]）绘制曲线：loss / mIoU+Dice / Acc / FWIoU+Kappa / lr"""
    if not history:
        return
    keys = history[0].keys()
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    def _plot(ax, names, title):
        for name in names:
            if name in keys:
                ys = [h.get(name) for h in history]
                ax.plot(epochs, ys, marker="o", markersize=3, label=name)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()

    _plot(axes[0], ["train_loss", "val_loss"], "Loss")
    _plot(axes[1], ["miou"], "mIoU")
    _plot(axes[2], ["mean_dice"], "Mean Dice")
    _plot(axes[3], ["pixel_acc", "mean_acc", "mean_precision"], "Accuracy / Precision")
    _plot(axes[4], ["fwiou", "kappa"], "FWIoU / Kappa")
    _plot(axes[5], ["lr"], "Learning Rate")
    # 去掉空图（如未验证）
    for ax in axes:
        if not ax.get_lines():
            ax.axis("off")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_preview_grid(samples, save_path, palette=None, titles=None):
    """数据集预览：每行一个样本（原图 | 标签 | 增强后输入）

    :param samples: list[(img_tensor_or_np, mask)]
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    titles = titles or ["Image", "Mask(GT)", "Augmented Input"]
    for row, (img, mask) in enumerate(samples):
        if isinstance(img, torch.Tensor) and img.dim() == 3 and img.shape[0] == 3:
            # 归一化后的张量：第三列直接展示（还原亮度便于观察）
            orig = denormalize(img)
            axes[row][2].imshow(orig)
            axes[row][0].imshow(orig)
        else:
            axes[row][0].imshow(img)
        m = np.asarray(mask)
        if palette is None:
            palette = palette_for(max(_mask_n_classes(m), 2))
        axes[row][1].imshow(colorize(m, palette))
        for c in range(3):
            if row == 0:
                axes[row][c].set_title(titles[c])
            axes[row][c].axis("off")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
