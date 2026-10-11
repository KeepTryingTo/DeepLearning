"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2026/10/11
@Desc   : COCO 数据集 -> 语义分割数据集转换工具。
          将 COCO 检测/实例分割标注（instances_xxx.json，多边形或 RLE）栅格化为
          语义分割索引 PNG（像素=类别id，0=背景），产出 images/ + masks/ + classes.txt，
          之后即可直接用 train_seg.py 训练（自动识别为自定义数据集）。

用法示例：
    python tools/coco2seg.py --ann annotations/instances_val2017.json \
        --img-dir val2017 --out-dir data/coco_seg
    # 只转部分类别：
    python tools/coco2seg.py --ann ... --img-dir ... --out-dir ... \
        --cats person car dog
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMG_EXTS = (".jpg", ".jpeg", ".png")


def parse_args():
    p = argparse.ArgumentParser(description="COCO -> 语义分割 mask 转换")
    p.add_argument("--ann", required=True, help="COCO 标注 json（instances_*.json）")
    p.add_argument("--img-dir", required=True, help="COCO 图片目录")
    p.add_argument("--out-dir", required=True, help="输出数据集目录")
    p.add_argument("--cats", nargs="*", default=None,
                   help="只保留指定类别名（默认全部类别）")
    p.add_argument("--skip-iscrowd", action="store_true", default=True,
                   help="跳过 iscrowd 的 RLE 标注（默认跳过）")
    p.add_argument("--copy-images", action="store_true", default=True,
                   help="复制图片到输出目录（默认开启）")
    return p.parse_args()


def decode_rle(rle, h, w):
    """需要 pycocotools；不可用时返回 None"""
    try:
        from pycocotools import mask as mask_utils
        return mask_utils.decode(rle)
    except Exception:
        return None


def main():
    args = parse_args()
    ann_path = Path(args.ann)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_images = out_dir / "images"
    out_masks = out_dir / "masks"
    out_masks.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        out_images.mkdir(parents=True, exist_ok=True)

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    # 类别重映射：category_id（稀疏）-> 连续 id 1..K（0 为背景）
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    if args.cats:
        keep = set(args.cats)
        cats = [c for c in cats if c["name"] in keep]
    cat2label = {c["id"]: i + 1 for i, c in enumerate(cats)}
    classes = ["background"] + [c["name"] for c in cats]
    print(f"[coco2seg] 类别: {classes}")

    imgs = {im["id"]: im for im in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in cat2label:
            continue
        if args.skip_iscrowd and ann.get("iscrowd", 0):
            continue
        if ann.get("segmentation") is None:
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    n_saved, n_skipped = 0, 0
    for img_id, im in imgs.items():
        anns = anns_by_img.get(img_id)
        if not anns:
            continue  # 无有效分割标注的图片跳过
        file_name = im["file_name"]
        src = img_dir / file_name
        if not src.exists():
            n_skipped += 1
            continue
        h, w = int(im["height"]), int(im["width"])
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for ann in anns:
            label = cat2label[ann["category_id"]]
            seg = ann["segmentation"]
            if isinstance(seg, list):
                # 多边形标注
                for poly in seg:
                    pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
                    if len(pts) >= 3:
                        draw.polygon(pts, fill=label)
            elif isinstance(seg, dict):
                # RLE 标注（需要 pycocotools）
                rle = seg.get("counts")
                if rle is None:
                    continue
                m = decode_rle(seg, h, w)
                if m is None:
                    print("[warn] RLE 解码失败（缺少 pycocotools），已跳过该标注")
                    continue
                arr = np.array(mask)
                arr[m > 0] = label
                mask = Image.fromarray(arr)
                draw = ImageDraw.Draw(mask)

        stem = Path(file_name).stem
        mask.save(str(out_masks / f"{stem}.png"))
        if args.copy_images:
            shutil.copy2(src, out_images / file_name)
        n_saved += 1

    (out_dir / "classes.txt").write_text(
        "\n".join(classes) + "\n", encoding="utf-8")
    print(f"[coco2seg] 完成: 转换 {n_saved} 张（跳过 {n_skipped} 张缺图）")
    print(f"[coco2seg] 输出目录: {out_dir}（images/ + masks/ + classes.txt）")
    print(f"[coco2seg] 使用 train_seg.py --dataset-root {out_dir} 训练即可")


if __name__ == "__main__":
    main()
