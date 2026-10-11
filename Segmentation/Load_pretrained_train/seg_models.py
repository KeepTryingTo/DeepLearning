"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2026/10/11
@Desc   : 模型工厂：加载 torchvision 任意预训练分割模型，并替换输出头以适配自定义类别数
          （迁移学习，与目标检测中替换 SSDLite 分类头的做法一致）
"""

from torch import nn

from config import SEG_MODELS, PRETRAIN_NUM_CLASSES


def _replace_last_conv(seq, num_classes):
    """将 Sequential 分类头中最后一个 Conv2d 替换为输出 num_classes 的新卷积。

    FCNHead:      [Conv, BN, ReLU, Dropout, Conv(1x1)]  -> 索引 4
    DeepLabHead:  [ASPP, Conv(1x1)]                     -> 索引 1
    统一按"最后一个 Conv2d"定位，兼容两种结构。
    """
    for i in range(len(seq) - 1, -1, -1):
        if isinstance(seq[i], nn.Conv2d):
            old = seq[i]
            seq[i] = nn.Conv2d(
                in_channels=old.in_channels,
                out_channels=num_classes,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                dilation=old.dilation,
                groups=old.groups,
                bias=True,
                padding_mode=old.padding_mode,
                device=old.weight.device,
                dtype=old.weight.dtype,
            )
            return True
    return False


def replace_heads(model, num_classes):
    """替换主分类头（FCN/DeepLabV3/LRASPP 三种头结构）与辅助分类头（如有）。

    返回被替换的模块名列表，便于日志确认。
    """
    replaced = []

    classifier = getattr(model, "classifier", None)
    if classifier is None:
        raise ValueError(f"模型 {type(model).__name__} 无 classifier 属性，无法替换分类头")

    if isinstance(classifier, nn.Sequential):
        # FCNHead / DeepLabHead
        if _replace_last_conv(classifier, num_classes):
            replaced.append("classifier")
    else:
        # LRASPPHead：低/高分辨率两个 1x1 分类卷积
        for attr in ("low_classifier", "high_classifier"):
            old = getattr(classifier, attr, None)
            if isinstance(old, nn.Conv2d):
                setattr(classifier, attr, nn.Conv2d(
                    old.in_channels, num_classes, kernel_size=1, bias=True,
                    device=old.weight.device, dtype=old.weight.dtype))
                replaced.append(f"classifier.{attr}")

    # 辅助头（FCN/DeepLabV3-ResNet 系列带有 aux_classifier，LRASPP/MobileNet 系列没有）
    aux = getattr(model, "aux_classifier", None)
    if isinstance(aux, nn.Sequential):
        if _replace_last_conv(aux, num_classes):
            replaced.append("aux_classifier")

    if not replaced:
        raise ValueError(f"模型 {type(model).__name__} 分类头替换失败")
    return replaced


def build_model(model_name, num_classes, pretrained=True):
    """构建分割模型（num_classes 含背景类）。

    :param model_name: config.SEG_MODELS 中的模型名
    :param num_classes: 目标类别数（含背景）
    :param pretrained: True 时加载 COCO-with-VOC-labels 预训练权重（含骨干），
                       False 时随机初始化（骨干权重也不下载）
    """
    if model_name not in SEG_MODELS:
        raise ValueError(
            f"未知分割模型: {model_name}，可选: {list(SEG_MODELS.keys())}")

    import torchvision.models.segmentation as tvm_seg
    fn = getattr(tvm_seg, model_name)

    if pretrained:
        # 在线加载预训练权重（首次运行会自动下载到 ~/.cache/torch/hub/checkpoints）
        model = fn(weights="DEFAULT")
    else:
        # weights_backbone=None 避免隐式下载骨干权重
        model = fn(weights=None, weights_backbone=None)

    # 预训练头是 21 类，迁移到自定义类别数时需要换头
    if num_classes != PRETRAIN_NUM_CLASSES:
        replaced = replace_heads(model, num_classes)
        print(f"[model] {model_name} 预训练头({PRETRAIN_NUM_CLASSES}类) "
              f"已替换为 {num_classes} 类: {replaced}")
    else:
        print(f"[model] {model_name} 类别数与预训练头一致({num_classes}类)，保留预训练头")
    return model


if __name__ == "__main__":
    # 简单自测：随机初始化构建各模型并替换头，检查前向输出形状
    import torch

    for name in SEG_MODELS:
        m = build_model(name, num_classes=5, pretrained=False)
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(1, 3, 96, 96))
        logits = out["out"]
        assert logits.shape == (1, 5, 96, 96), f"{name}: {logits.shape}"
        print(f"  -> {name} forward ok: {tuple(logits.shape)}")
