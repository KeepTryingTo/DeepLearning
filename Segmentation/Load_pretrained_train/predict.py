"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2026/10/11
@Desc   : 使用训练产物（best.pth / last.pth）进行分割推理与可视化：
          保存彩色 mask、叠加效果图，并打印各类像素占比。

用法示例：
    python predict.py --weight outputs/fcn_resnet_xxx/best.pth --source test_dir
    python predict.py --weight outputs/xxx/best.pth --source img.jpg --alpha 0.6
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import IMAGENET_MEAN, IMAGENET_STD, palette_for
from seg_models import build_model
import seg_visualize as vis

_RESAMPLE = getattr(Image, "Resampling", Image)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    p = argparse.ArgumentParser(description="分割模型推理与可视化")
    p.add_argument("--weight", required=True, help="训练产物的 checkpoint 路径")
    p.add_argument("--source", required=True, help="单张图片或图片目录")
    p.add_argument("--save-dir", default="runs/predict", help="可视化结果保存目录")
    p.add_argument("--alpha", type=float, default=0.5, help="叠加透明度")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--img-size", type=int, default=None,
                   help="推理输入边长（默认使用训练时的 img_size，保存在 checkpoint 中）")
    return p.parse_args()


def collect_images(source):
    src = Path(source)
    if src.is_file():
        return [src]
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise ValueError(f"目录中没有图片: {src}")
    return imgs


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))

    # 从 checkpoint 恢复模型与元信息
    ckpt = torch.load(args.weight, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model", "fcn_resnet50")
    classes = ckpt.get("classes") or [f"class_{i}" for i in range(ckpt["num_classes"])]
    num_classes = len(classes)
    img_size = args.img_size or int(ckpt.get("img_size", 480))
    palette = palette_for(num_classes)

    model = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"[predict] model={model_name} | classes={classes} | img_size={img_size} "
          f"| epoch={ckpt.get('epoch')} | best_miou={ckpt.get('best_miou')}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    stats_all = {}
    for img_path in collect_images(args.source):
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        # 与训练一致的输入尺寸；模型内部会将 logits 插值回输入尺寸
        x = image.resize((img_size, img_size), _RESAMPLE.BILINEAR)
        x = preprocess(x).unsqueeze(0).to(device)
        out = model(x)
        logits = out["out"] if isinstance(out, dict) else out
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        # 上采样回原图分辨率
        pred_full = np.array(Image.fromarray(pred).resize((W, H), _RESAMPLE.NEAREST))

        # 可视化：彩色 mask / 叠加 / 对比图
        stem = img_path.stem
        color_mask = vis.colorize(pred_full, palette)
        overlay = vis.overlay(np.array(image), color_mask, alpha=args.alpha)
        Image.fromarray(color_mask).save(str(save_dir / f"{stem}_mask.png"))
        Image.fromarray(overlay).save(str(save_dir / f"{stem}_overlay.png"))
        vis.save_comparison(np.array(image), pred_full, pred_full,
                            save_dir / f"{stem}_compare.png", palette=palette,
                            alpha=args.alpha)

        # 各类像素占比
        total = pred_full.size
        stats = {}
        for c in np.unique(pred_full):
            name = classes[c] if c < len(classes) else f"class_{c}"
            if name == "background":
                continue
            stats[name] = round(int((pred_full == c).sum()) / total, 4)
        stats_all[img_path.name] = stats
        print(f"[{img_path.name}] 各类像素占比: {stats}")

    (save_dir / "pixel_stats.json").write_text(
        json.dumps(stats_all, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[predict] 结果已保存到: {save_dir}")


if __name__ == "__main__":
    main()
