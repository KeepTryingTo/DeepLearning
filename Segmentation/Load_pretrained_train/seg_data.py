"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2026/10/11
@Desc   : 分割数据集加载：自动识别 PASCAL VOC / 自定义（Train-Test 子目录 或 images+masks 扁平）
          以及 COCO 转换结果（tools/coco2seg.py 产出的 images/+masks/+classes.txt）。

支持的数据集目录格式：
1. PASCAL VOC 官方结构（自动下探 VOCdevkit/VOC20xx）：
   root/VOCdevkit/VOC2012/
   ├── JPEGImages/            # 原图
   ├── SegmentationClass/     # 索引 PNG 标签（像素=类别id，255=忽略）
   └── ImageSets/Segmentation/{train,val}.txt

2. Train/Test 子目录结构：
   root/Train/{Images|masks|images|JPEGImages|SegmentationClass|...}
   root/Test/（或 Val/）

3. 扁平结构（COCO 转换结果或自定义）：
   root/images/*.jpg + root/masks/*.png
   可选 root/classes.txt（每行一个类别名，第 1 行=背景）
   可选 root/ImageSets/Segmentation/{train,val}.txt 划分，否则按 seed=42 自动 8:2 划分
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import (IMAGENET_MEAN, IMAGENET_STD, IGNORE_INDEX, VOC_CLASSES)

# 兼容 Pillow 版本（Pillow>=10 移除了 Image.BILINEAR 常量）
_RESAMPLE = getattr(Image, "Resampling", Image)
BILINEAR = _RESAMPLE.BILINEAR
NEAREST = _RESAMPLE.NEAREST

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

_IMG_DIR_NAMES = ("images", "Images", "JPEGImages", "imgs", "img")
_MASK_DIR_NAMES = ("masks", "Masks", "SegmentationClass", "labels", "gt")


def _find_dir(root: Path, names):
    for name in names:
        d = root / name
        if d.is_dir():
            return d
    return None


def _find_image(img_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    return None


def _read_classes_file(path: Path):
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    return lines or None


class SegDataset(Dataset):
    """索引 PNG 标签数据集。mask 像素值 = 类别 id，255 = 忽略。"""

    def __init__(self, root, img_size=480, train=True, augment=True,
                 classes_file=None, num_classes=None):
        """
        :param root: 数据集根目录（自动识别上述三种结构）
        :param img_size: 训练/评估时统一缩放的边长（正方形）
        :param train: True 读取训练划分，False 读取验证/测试划分
        :param augment: 训练集数据增强（水平翻转 + 轻度颜色扰动）
        :param classes_file: 显式指定 classes.txt（默认自动查找）
        :param num_classes: 显式指定类别数（默认由 classes.txt 或标签推断）
        """
        self.root = Path(root)
        self.img_size = int(img_size)
        self.train = train
        self.augment = bool(augment and train)

        # VOCdevkit 标准数据集（root/VOCdevkit/VOC20xx/...）自动下探
        if (self.root / "VOCdevkit").is_dir():
            for vr in sorted((self.root / "VOCdevkit").iterdir()):
                if (vr / "JPEGImages").is_dir():
                    self.root = vr
                    break

        # 识别 Train/Test 子目录结构
        split_dir = None
        for s in (["Train", "train"] if train else ["Test", "test", "Val", "val"]):
            if (self.root / s).is_dir():
                split_dir = self.root / s
                break

        base = split_dir if split_dir is not None else self.root
        img_dir = _find_dir(base, _IMG_DIR_NAMES)
        mask_dir = _find_dir(base, _MASK_DIR_NAMES)
        if img_dir is None or mask_dir is None:
            raise ValueError(
                f"数据集目录需包含 images/(或 JPEGImages) 与 masks/(或 SegmentationClass): {root}")
        self.img_dir, self.mask_dir = img_dir, mask_dir

        # ---------- 类别名 ----------
        self.classes = None
        if classes_file is not None:
            self.classes = _read_classes_file(Path(classes_file))
        if self.classes is None:
            for cand_base in (self.root, split_dir):
                if cand_base is None:
                    continue
                cf = cand_base / "classes.txt"
                if cf.exists():
                    self.classes = _read_classes_file(cf)
                    break
        if self.classes is None and (self.root / "JPEGImages").is_dir():
            # VOC 标准结构且未提供 classes.txt 时使用 VOC 21 类
            self.classes = list(VOC_CLASSES)

        # ---------- 样本划分 ----------
        stems = None
        if split_dir is None:
            isd = self.root / "ImageSets" / "Segmentation"
            split = "train" if train else "val"
            for name in (split, {"train": "trainval", "val": "test"}.get(split, "")):
                f = isd / f"{name}.txt"
                if f.exists():
                    stems = [l.strip() for l in f.read_text().splitlines() if l.strip()]
                    break
        if stems is None:
            png_stems = sorted(p.stem for p in mask_dir.iterdir()
                               if p.suffix.lower() == ".png")
            if not png_stems:
                raise ValueError(f"masks 目录无 PNG 标签: {mask_dir}")
            if split_dir is not None:
                # Train/Test 模式：使用该子目录全部样本
                stems = png_stems
            else:
                # 无划分文件：按固定种子 8:2 自动划分（训练/验证一致可复现）
                g = np.random.RandomState(42)
                order = g.permutation(len(png_stems))
                n_val = max(1, int(len(png_stems) * 0.2))
                idx = order[n_val:] if train else order[:n_val]
                stems = [png_stems[i] for i in idx]

        self.items = []
        for stem in stems:
            img = _find_image(img_dir, stem)
            mask = mask_dir / (stem + ".png")
            if img is not None and mask.exists():
                self.items.append((img, mask))
        if not self.items:
            raise ValueError(f"未找到配对的 image/mask 样本 (root={root}, train={train})")

        # ---------- 类别数 ----------
        if num_classes is not None:
            self.num_classes = int(num_classes)
        elif self.classes:
            self.num_classes = len(self.classes)
        else:
            self.num_classes = self._infer_num_classes()
        if self.classes is not None and len(self.classes) != self.num_classes:
            raise ValueError(
                f"classes.txt 类别数({len(self.classes)}) 与标签推断类别数"
                f"({self.num_classes}) 不一致，请检查数据或用 --num-classes 显式指定")

        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)

        tag = "train" if train else "val"
        print(f"[dataset] {tag}: {len(self.items)} 对样本 | "
              f"img_dir={img_dir} | mask_dir={mask_dir} | "
              f"num_classes={self.num_classes}")

    def _infer_num_classes(self):
        max_id = 0
        for _, mask in self.items:
            arr = np.array(Image.open(mask))
            valid = arr[arr != IGNORE_INDEX]
            if valid.size:
                max_id = max(max_id, int(valid.max()))
        return max_id + 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if mask.mode == "RGB":
            raise ValueError(
                f"mask 必须是索引(灰度/调色板) PNG 而非 RGB: {mask_path}")

        s = self.img_size
        img = img.resize((s, s), BILINEAR)
        mask = mask.resize((s, s), NEAREST)

        # 同步数据增强：水平翻转 + 仅作用于图像的颜色扰动
        if self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img = self.color_jitter(img)

        img_t = self.transform_img(img)
        mask_t = torch.as_tensor(np.array(mask), dtype=torch.int64)
        # 清洗越界标签：不属于 [0, num_classes-1] 且非忽略索引的像素置为忽略
        invalid = (mask_t >= self.num_classes) & (mask_t != IGNORE_INDEX)
        if invalid.any():
            mask_t[invalid] = IGNORE_INDEX
        return img_t, mask_t


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    for train in (True, False):
        ds = SegDataset(root, img_size=480, train=train)
        if train:
            img, mask = ds[0]
            print(f"sample: img {tuple(img.shape)} {img.dtype} | "
                  f"mask {tuple(mask.shape)} {mask.dtype} | "
                  f"labels={torch.unique(mask).tolist()}")
