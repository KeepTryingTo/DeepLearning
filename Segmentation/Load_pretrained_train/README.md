# Load_pretrained_train —— torchvision 预训练分割模型迁移学习训练

对标 `ObjectDetection/.../transferLearningObjectDetect_SSDLite.py` 的目标检测迁移学习模式，
将其迁移到**语义分割**：加载 torchvision 任意 COCO-with-VOC-labels 预训练分割模型
（FCN / DeepLabV3 / LRASPP 系列骨干），替换输出头适配自定义类别数，在自己的数据集上微调。

## 功能特性

- **任意预训练模型**：`fcn_resnet50 / fcn_resnet101 / deeplabv3_resnet50 / deeplabv3_resnet101 /
  deeplabv3_mobilenet_v3_large / lraspp_mobilenet_v3_large`，自动识别 FCNHead / DeepLabHead /
  LRASPPHead 三种头结构并替换为自定义类别数（含 aux 辅助头）
- **三种数据集格式**（自动识别）：
  - PASCAL VOC 官方结构（`VOCdevkit/VOC20xx/JPEGImages + SegmentationClass + ImageSets`）
  - COCO（先用 `tools/coco2seg.py` 把 instances json 栅格化为 mask，再训练）
  - 自定义：`Train/Images+masks` 子目录 或 `images/+masks/` 扁平结构（无划分文件时按 8:2 自动划分）
- **完整指标**：Pixel Acc / Mean Acc / Mean Precision / **mIoU** / FWIoU / Mean Dice / Kappa
  以及逐类 IoU、Dice、Precision、Recall（混淆矩阵累积，255 忽略像素不计）
- **自定义损失**：`ce`、`dice`、`focal`（可调 gamma）、`ce_dice`、`focal_dice`，
  统一处理模型 dict 输出（out + aux 辅助头，aux 权重可配）
- **可视化**：验证时的 原图|标签|预测|叠加 四联对比图、训练曲线（loss/mIoU/Dice/Acc/lr）、
  数据集预览网格（`tools/preview_data.py`）、推理彩色 mask + 叠加图 + 各类像素占比
- **训练配方**：poly/step/cos 学习率调度、aux 头 10 倍学习率（DeepLab 官方配方）、
  SGD/Adam/AdamW、断点续训（`--resume`）、早停（`--early-stop`）、可选 TensorBoard

## 目录结构

```
Load_pretrained_train/
├── config.py           # 常量：调色板、VOC 类别、模型注册表
├── seg_data.py         # SegDataset：VOC / 自定义数据自动识别
├── seg_models.py       # 模型工厂：任意预训练分割模型换头
├── seg_losses.py       # 损失：CE / Dice / Focal / 组合
├── seg_metrics.py      # 混淆矩阵 + 全套指标
├── seg_visualize.py    # 上色 / 叠加 / 对比图 / 曲线
├── train_seg.py        # 训练主脚本
├── predict.py          # 推理 + 可视化
└── tools/
    ├── coco2seg.py     # COCO json -> 语义 mask PNG
    └── preview_data.py # 数据集预览
```

## 快速开始

### 1. 准备数据集

任选一种：

**自定义 / COCO 转换（扁平结构）**

```
data/mydata/
├── images/       # *.jpg / *.png
├── masks/        # *.png，像素值=类别id（0=背景），255=忽略
└── classes.txt   # 可选：每行一个类别名，第 1 行=背景
```

无 `ImageSets/Segmentation/train|val.txt` 划分文件时自动 8:2 划分（seed=42 可复现）。

**PASCAL VOC 官方结构**：直接把 `VOCdevkit` 所在目录作为 `--dataset-root`，
会自动下探 `VOCdevkit/VOC20xx` 并使用 `ImageSets/Segmentation` 的官方划分与 VOC 21 类名。

**COCO**：先转换再训练

```bash
python tools/coco2seg.py --ann annotations/instances_val2017.json \
    --img-dir val2017 --out-dir data/coco_seg
# 只保留部分类别：--cats person car dog
```

### 2. 预览数据（建议）

```bash
python tools/preview_data.py --dataset-root data/mydata --num 8
```

### 3. 训练

```bash
# FCN-ResNet50 + CE+Dice 损失 + poly 调度（分割标准配方）
python train_seg.py --model fcn_resnet50 --dataset-root data/mydata \
    --loss ce_dice --epochs 30 --batch-size 4 --img-size 480

# DeepLabV3-ResNet101 在 VOC 上微调
python train_seg.py --model deeplabv3_resnet101 --dataset-root data/VOCdevkit \
    --loss ce --optimizer sgd --lr 0.007 --scheduler poly --epochs 50

# 断点续训
python train_seg.py --model fcn_resnet50 --dataset-root data/mydata \
    --resume outputs/fcn_resnet50_xxx/last.pth --epochs 60
```

首次运行会自动下载预训练权重到 `~/.cache/torch/hub/checkpoints`。

### 4. 推理与可视化

```bash
python predict.py --weight outputs/fcn_resnet50_xxx/best.pth --source test_dir
```

输出 `{stem}_mask.png`（彩色 mask）、`{stem}_overlay.png`（叠加效果）、
`{stem}_compare.png`（对比图）与 `pixel_stats.json`（各类像素占比）。

## 训练输出说明（outputs/<model>_<时间戳>/）

| 文件 | 说明 |
|---|---|
| `best.pth` / `last.pth` | 最优（按 mIoU）/ 最新 checkpoint，含模型名、类别、img_size、优化器状态 |
| `metrics.csv` | 每 epoch 的 loss 与全部指标 |
| `per_class_metrics.csv` | 逐类 IoU / Dice / P / R / 支持像素数 |
| `curves.png` | 训练曲线（loss、mIoU、Dice、Acc、FWIoU/Kappa、lr） |
| `vis/epoch*_sample*.png` | 验证集预测对比图（原图\|标签\|预测\|叠加） |
| `classes.json` / `history.json` | 类别名 / 完整训练历史 |
| `tb/` | TensorBoard 日志（安装 tensorboard 后自动启用） |

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | `fcn_resnet50` | 预训练模型名（见 config.SEG_MODELS） |
| `--dataset-root` | 必填 | 数据集根目录 |
| `--img-size` | 480 | 训练输入边长（正方形） |
| `--loss` | `ce` | `ce / dice / focal / ce_dice / focal_dice` |
| `--optimizer` | `sgd` | `sgd / adam / adamw` |
| `--scheduler` | `poly` | `poly / step / cos / none` |
| `--lr` | 1e-3 | 基础学习率（aux 头自动 ×10） |
| `--aux-weight` | 0.5 | 辅助头损失权重（0=不使用） |
| `--aux-lr-mult` | 10 | 辅助头学习率倍数 |
| `--no-pretrained` | 关 | 随机初始化（不加载预训练权重） |
| `--early-stop` | 0 | 早停耐心轮数（0=关闭） |
| `--resume` | 无 | 断点续训 checkpoint |

## 注意事项

- mask 必须是**索引/灰度 PNG**（像素值=类别 id），不能是 RGB 三通道图；
- 类别数 = `classes.txt` 行数 = mask 中最大类别 id + 1（背景占 id 0）；
- Windows 下 `--workers` 建议 0（>0 时脚本已用 `if __name__ == "__main__"` 保护）；
- DeepLabV3 训练时若某批恰好只剩 1 个样本，ASPP 池化分支的 BN 会报错
  （`Expected more than 1 value per channel`），脚本已自动在该情形下 `drop_last`；
  若数据集极小，请保证 `样本数 > batch_size` 或调小 `--batch-size`；
- 标签中的 255 像素在损失与指标中均被忽略（VOC 边界惯例）；
- torchvision 预训练权重为 COCO-with-VOC-labels 21 类输出，自定义类别数时换头后仅骨干保留预训练。
