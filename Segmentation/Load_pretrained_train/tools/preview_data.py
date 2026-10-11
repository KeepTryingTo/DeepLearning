"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2026/10/11
@Desc   : 数据集预览工具：保存若干样本的 原图|标签|增强后输入 网格图，
          训练前检查数据读取与增强是否正确。

用法示例：
    python tools/preview_data.py --dataset-root data/mydata --num 8
    python tools/preview_data.py --dataset-root data/mydata --split val
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# 以 python tools/preview_data.py 方式运行时，把项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import palette_for
from seg_data import SegDataset
import seg_visualize as vis


def parse_args():
    p = argparse.ArgumentParser(description="分割数据集预览")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--img-size", type=int, default=480)
    p.add_argument("--num", type=int, default=8, help="预览样本数")
    p.add_argument("--no-augment", action="store_true", help="关闭增强效果展示")
    p.add_argument("--save-path", default="runs/preview.png")
    p.add_argument("--classes-file", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    ds = SegDataset(args.dataset_root, img_size=args.img_size,
                    train=(args.split == "train"),
                    augment=(args.split == "train") and not args.no_augment,
                    classes_file=args.classes_file)
    if len(ds) == 0:
        raise ValueError("数据集为空")

    idxs = np.linspace(0, len(ds) - 1, num=min(args.num, len(ds))).astype(int).tolist()
    samples = []
    for i in idxs:
        img_t, mask_t = ds[i]
        # 原图（未增强）单独读取一份用于对照
        img_path, mask_path = ds.items[i]
        orig = np.array(Image.open(img_path).convert("RGB").resize(
            (args.img_size, args.img_size), Image.Resampling.BILINEAR
            if hasattr(Image, "Resampling") else Image.BILINEAR))
        samples.append((orig, mask_t.numpy()))

    palette = palette_for(ds.num_classes)
    vis.save_preview_grid(samples, args.save_path, palette=palette,
                          titles=[f"Image", f"Mask (num_classes={ds.num_classes})",
                                  "Augmented Input"])
    print(f"[preview] 已保存 {len(samples)} 个样本的预览图: {args.save_path}")
    print(f"[preview] 类别数: {ds.num_classes} | "
          f"类别: {ds.classes or '（未提供 classes.txt，由标签推断）'}")


if __name__ == "__main__":
    main()
