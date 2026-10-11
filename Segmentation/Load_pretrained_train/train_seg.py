"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2026/10/11
@Desc   : 加载 torchvision 任意预训练分割模型进行迁移学习训练（对标目标检测中的
          transferLearningObjectDetect_SSDLite.py：加载预训练权重 + 替换分类头 + 自定义数据训练）。

          支持：PASCAL VOC / COCO（经 tools/coco2seg.py 转换）/ 自定义数据集；
                ce|dice|focal|ce_dice|focal_dice 损失；
                完整指标（mIoU/mDice/pAcc/mAcc/P/R/FWIoU/Kappa + 逐类指标）；
                预测可视化、训练曲线、断点续训、早停。

用法示例：
    python train_seg.py --model fcn_resnet50 --dataset-root data/mydata --epochs 30
    python train_seg.py --model deeplabv3_resnet101 --dataset-root data/VOCdevkit \
        --loss ce_dice --optimizer sgd --scheduler poly --batch-size 4
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (IGNORE_INDEX, LOSS_OPTIONS, OPTIMIZER_OPTIONS,
                    SCHEDULER_OPTIONS, SEG_MODELS, palette_for)
from seg_data import SegDataset
from seg_losses import SegCriterion
from seg_metrics import SegMetrics
from seg_models import build_model
import seg_visualize as vis

CSV_HEADER = ["epoch", "lr", "train_loss", "val_loss", "pixel_acc", "mean_acc",
              "mean_precision", "miou", "fwiou", "mean_dice", "kappa"]


def parse_args():
    p = argparse.ArgumentParser(description="torchvision 分割模型迁移学习训练")
    # 模型与数据
    p.add_argument("--model", default="fcn_resnet50", choices=list(SEG_MODELS.keys()),
                   help="预训练分割模型名")
    p.add_argument("--dataset-root", required=True, help="数据集根目录（VOC/自定义自动识别）")
    p.add_argument("--classes-file", default=None, help="classes.txt 路径（默认自动查找）")
    p.add_argument("--num-classes", type=int, default=None,
                   help="显式指定类别数（含背景；默认由数据推断）")
    p.add_argument("--img-size", type=int, default=480, help="训练输入边长（正方形）")
    p.add_argument("--no-augment", action="store_true", help="关闭训练数据增强")
    # 训练超参
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=0, help="DataLoader 进程数（Windows 建议 0）")
    p.add_argument("--optimizer", default="sgd", choices=OPTIMIZER_OPTIONS)
    p.add_argument("--lr", type=float, default=1e-3, help="基础学习率")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9, help="SGD 动量")
    p.add_argument("--scheduler", default="poly", choices=SCHEDULER_OPTIONS,
                   help="学习率调度：poly（分割标准）|step|cos|none")
    # 损失
    p.add_argument("--loss", default="ce", choices=LOSS_OPTIONS)
    p.add_argument("--aux-weight", type=float, default=0.5,
                   help="辅助头损失权重（0 表示不使用 aux 头损失）")
    p.add_argument("--aux-lr-mult", type=float, default=10.0,
                   help="辅助头学习率倍数（DeepLab 训练配方为 10）")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    # 运行控制
    p.add_argument("--no-pretrained", action="store_true",
                   help="不加载预训练权重（随机初始化）")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--output-dir", default=None,
                   help="输出目录（默认 outputs/<model>_<时间戳>）")
    p.add_argument("--val-interval", type=int, default=1, help="每 N 个 epoch 验证一次")
    p.add_argument("--vis-num", type=int, default=4, help="每次验证保存的预测对比图数量")
    p.add_argument("--log-interval", type=int, default=20, help="每 N 个 step 打印日志")
    p.add_argument("--early-stop", type=int, default=0, help="早停耐心轮数（0=关闭）")
    p.add_argument("--resume", default=None, help="断点续训的 checkpoint 路径")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] 指定 cuda 但不可用，回退到 cpu")
        return torch.device("cpu")
    return torch.device(name)


def build_optimizer(model, args):
    """构造优化器：辅助头参数使用 aux_lr_mult 倍学习率（DeepLab 训练配方）"""
    base_params, aux_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (aux_params if "aux_classifier" in n else base_params).append(p)
    if aux_params and args.aux_lr_mult != 1.0:
        groups = [{"params": base_params},
                  {"params": aux_params, "lr": args.lr * args.aux_lr_mult}]
    else:
        groups = [{"params": base_params + aux_params}]

    if args.optimizer == "adam":
        return torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum,
                           weight_decay=args.weight_decay, nesterov=True)


def build_scheduler(optimizer, args):
    """poly 按步手动衰减（训练循环内处理）；step/cos 交给 torch 调度器按 epoch 衰减"""
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)
    if args.scheduler == "cos":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs))
    return None


def save_checkpoint(path, model, optimizer, epoch, best_miou, args, classes, metrics):
    torch.save({
        "model": args.model,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_miou": best_miou,
        "num_classes": len(classes),
        "classes": classes,
        "img_size": args.img_size,
        "loss": args.loss,
        "metrics": metrics,
    }, str(path))


def train_one_epoch(model, loader, criterion, optimizer, device, args,
                    epoch, step_offset, total_steps, writer):
    model.train()
    total_loss, n_steps = 0.0, 0
    # 各参数组的基础学习率（组1为辅助头，按 aux_lr_mult 放大）
    base_lrs = [args.lr * (args.aux_lr_mult if (i == 1 and len(optimizer.param_groups) > 1
                                                and args.aux_lr_mult != 1.0) else 1.0)
                for i in range(len(optimizer.param_groups))]

    for imgs, masks in loader:
        step = step_offset + n_steps
        # 学习率调度
        if args.scheduler == "poly":
            factor = (1.0 - min(step, total_steps) / max(1, total_steps)) ** 0.9
            for g, b in zip(optimizer.param_groups, base_lrs):
                g["lr"] = b * factor
        elif args.scheduler == "none":
            for g, b in zip(optimizer.param_groups, base_lrs):
                g["lr"] = b

        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1
        if n_steps % args.log_interval == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"[epoch {epoch}] step {n_steps}/{len(loader)} | "
                  f"loss {total_loss / n_steps:.4f} | lr {cur_lr:.6f}")
            if writer is not None:
                writer.add_scalar("train/loss_step", loss.item(),
                                  global_step=step_offset + n_steps)
    return total_loss / max(1, n_steps), n_steps


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, args,
             save_dir=None, palette=None, epoch=None):
    """验证：返回 (val_loss, SegMetrics)；可选保存预测对比图"""
    model.eval()
    meter = SegMetrics(num_classes, ignore_index=IGNORE_INDEX)
    total_loss, n_batches = 0.0, 0
    vis_saved = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        outputs = model(imgs)
        logits = outputs["out"] if isinstance(outputs, dict) else outputs
        total_loss += criterion(outputs, masks).item()
        n_batches += 1
        pred = logits.argmax(dim=1).cpu()
        meter.update(pred.numpy(), masks.cpu().numpy())

        # 保存若干张 原图|标签|预测|叠加 对比图
        if save_dir is not None and vis_saved < args.vis_num:
            for i in range(imgs.shape[0]):
                if vis_saved >= args.vis_num:
                    break
                vis.save_comparison(
                    imgs[i].cpu(), masks[i].cpu(), pred[i],
                    Path(save_dir) / f"epoch{epoch}_sample{vis_saved}.png",
                    palette=palette)
                vis_saved += 1

    val_loss = total_loss / max(1, n_batches)
    return val_loss, meter


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir \
        else Path("outputs") / f"{args.model}_{stamp}"
    vis_dir = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir}")

    # 可选 TensorBoard（未安装则跳过，不影响训练）
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
        print("[tensorboard] 已启用")
    except Exception as e:
        print(f"[tensorboard] 不可用，跳过（{e.__class__.__name__}）")

    # ---------------- 数据 ----------------
    print('-----------------------------------------加载数据-----------------------------------------')
    train_ds = SegDataset(args.dataset_root, img_size=args.img_size, train=True,
                          augment=not args.no_augment, classes_file=args.classes_file,
                          num_classes=args.num_classes)
    val_ds = SegDataset(args.dataset_root, img_size=args.img_size, train=False,
                        augment=False, classes_file=args.classes_file,
                        num_classes=train_ds.num_classes)
    num_classes = train_ds.num_classes
    classes = train_ds.classes or [f"class_{i}" for i in range(num_classes)]
    palette = palette_for(num_classes)
    (out_dir / "classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[data] 类别数(含背景): {num_classes} | 类别: {classes}")

    # 注意：DeepLabV3 的 ASPP 池化分支含 BN，batch=1 时会报
    # "Expected more than 1 value per channel"，因此最后一批恰好剩 1 个样本时丢弃
    drop_last = len(train_ds) > args.batch_size and len(train_ds) % args.batch_size == 1
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ---------------- 模型 / 损失 / 优化器 ----------------
    print('-----------------------------------------构建模型-----------------------------------------')
    model = build_model(args.model, num_classes, pretrained=not args.no_pretrained)
    model.to(device)
    criterion = SegCriterion(num_classes, loss_name=args.loss,
                             ignore_index=IGNORE_INDEX, aux_weight=args.aux_weight,
                             focal_gamma=args.focal_gamma)
    optimizer = build_optimizer(model, args)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    torch_scheduler = build_scheduler(optimizer, args)
    print(f"[train] loss={args.loss} | optimizer={args.optimizer} lr={args.lr} | "
          f"scheduler={args.scheduler} | epochs={args.epochs} | batch={args.batch_size}")

    # ---------------- 断点续训 ----------------
    start_epoch, best_miou, history = 1, -1.0, []
    step_offset = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_miou = float(ckpt.get("best_miou", -1.0))
        step_offset = int((start_epoch - 1) * steps_per_epoch)
        hist_file = Path(args.resume).parent / "history.json"
        if hist_file.exists():
            history = json.loads(hist_file.read_text(encoding="utf-8"))
        print(f"[resume] 从 epoch {start_epoch} 继续（best_miou={best_miou:.4f}）")

    with open(out_dir / "metrics.csv", "a", newline="", encoding="utf-8") as f:
        if start_epoch == 1:
            csv.writer(f).writerow(CSV_HEADER)

    # ---------------- 训练循环 ----------------
    best_path, last_path = out_dir / "best.pth", out_dir / "last.pth"
    no_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        print(f'--------------------------------------第 {epoch}/{args.epochs} 轮训练'
              f'--------------------------------------')
        train_loss, n_steps = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args,
            epoch, step_offset, total_steps, writer)
        step_offset += n_steps
        if torch_scheduler is not None:
            torch_scheduler.step()

        record = {"epoch": epoch,
                  "lr": optimizer.param_groups[0]["lr"],
                  "train_loss": round(train_loss, 6)}
        print(f"[epoch {epoch}] train_loss {train_loss:.4f}")

        # ---------------- 验证 ----------------
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            val_loss, meter = evaluate(
                model, val_loader, criterion, device, num_classes, args,
                save_dir=vis_dir, palette=palette, epoch=epoch)
            overall, per_class = meter.compute()
            print(f"[epoch {epoch}] val_loss {val_loss:.4f}")
            print(meter.summary_str(classes))

            record.update({"val_loss": round(val_loss, 6),
                           **{k: round(v, 6) for k, v in overall.items()}})
            if writer is not None:
                writer.add_scalar("val/loss", val_loss, global_step=epoch)
                for k, v in overall.items():
                    writer.add_scalar(f"val/{k}", v, global_step=epoch)

            # 保存最优 / 最新模型
            is_best = overall["miou"] > best_miou
            if is_best:
                best_miou = overall["miou"]
                no_improve = 0
                print(f"  -> 保存最优模型 (mIoU={best_miou:.4f})")
                save_checkpoint(best_path, model, optimizer, epoch, best_miou,
                                args, classes, overall)
            else:
                no_improve += 1
            save_checkpoint(last_path, model, optimizer, epoch, best_miou,
                            args, classes, overall)

            # 逐类指标 CSV（每轮覆盖，最终为最优/最后一轮结果）
            with open(out_dir / "per_class_metrics.csv", "w",
                      newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["class", "iou", "dice", "precision", "recall", "support"])
                for c, m in enumerate(per_class):
                    w.writerow([classes[c] if c < len(classes) else f"class_{c}",
                                f"{m['iou']:.6f}" if np.isfinite(m["iou"]) else "",
                                f"{m['dice']:.6f}" if np.isfinite(m["dice"]) else "",
                                f"{m['precision']:.6f}" if np.isfinite(m["precision"]) else "",
                                f"{m['recall']:.6f}" if np.isfinite(m["recall"]) else "",
                                m["support"]])

            # 早停（以 mIoU 为准）
            if args.early_stop > 0 and no_improve >= args.early_stop:
                print(f"[early-stop] mIoU 已 {args.early_stop} 轮未提升，提前停止")
                break

        history.append(record)
        # 增量落盘：metrics.csv / history.json / 曲线图
        with open(out_dir / "metrics.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([record.get(k, "") for k in CSV_HEADER])
        (out_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        vis.plot_history(history, out_dir / "curves.png")
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, global_step=epoch)
            writer.add_scalar("train/lr", record["lr"], global_step=epoch)

    # ---------------- 收尾 ----------------
    if writer is not None:
        writer.close()
    vis.plot_history(history, out_dir / "curves.png")
    print('-----------------------------------------训练结束-----------------------------------------')
    print(f"最优 mIoU: {best_miou:.4f}")
    print(f"最优权重: {best_path}\n最新权重: {last_path}")
    print(f"曲线图: {out_dir / 'curves.png'} | 逐类指标: {out_dir / 'per_class_metrics.csv'}")


if __name__ == "__main__":
    main()
