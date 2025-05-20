"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/5/19-20:42
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
from typing import Any, Optional

import torch
import torch.optim as optim
from pytorch_lightning.utilities.types import STEP_OUTPUT, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import models, datasets, ops
from torchvision.transforms import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import pytorch_lightning as L

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Use a pretrained Faster R-CNN model from torchvision and modify it
class FaceDetectionModel(L.LightningModule):
    def __init__(self, lr = 0.001, momentum = 0.9, weight_decay = 0.0005):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.model = ssdlite320_mobilenet_v3_large(pretrained=True,progress=True)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        imgs, annot = batch
        images, targets = FacesData.convert_inputs(imgs, annot, device=self.device)
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses)
        return losses

    def validation_step(self, batch, batch_idx):
        imgs, annot = batch
        images, targets = FacesData.convert_inputs(imgs, annot, device=self.device)
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('val_loss', losses)
        return losses

    def configure_optimizers(self):
        return optim.SGD(
            params=self.parameters(), lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

def transform(data):
    img, boxes = data["image"], data["boxes"]
    # TODO (height, width)
    img_size = img.size
    original_size = img_size[1], img_size[0]

    def scale_boxes(boxes, original_size, new_size):
        h_ratio = new_size[0] / original_size[0]
        w_ratio = new_size[1] / original_size[1]
        boxes = boxes * torch.tensor([w_ratio, h_ratio, w_ratio, h_ratio]).to(device)
        return boxes

    img = transforms.ToTensor()(img)
    img = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(img)
    img = transforms.Resize(size=(800,800))(img)
    new_size = (800,800)
    # 调整 boxes（如果需要）
    boxes = scale_boxes(boxes, original_size, new_size)

    return {"image": img, "boxes": boxes}

class FacesData(L.LightningDataModule):
    def __init__(self, root):
        super().__init__()
        self.root = root

    @staticmethod
    def convert_inputs(imgs, annot, device, small_thr=0.001):
        """将数据集转换为可以接受的格式"""
        images, targets = [], []
        for img, annot in zip(imgs, annot):
            bbox = annot['bbox']
            small = (bbox[:, 2] * bbox[:, 3]) <= (img.size[1] * img.size[0] * small_thr)
            boxes = ops.box_convert(bbox[~small], in_fmt='xywh', out_fmt='xyxy')
            output_dict = transform({"image": img, "boxes": boxes})
            images.append(output_dict['image'].to(device))
            targets.append({
                'boxes': output_dict['boxes'].to(device),
                'labels': torch.ones(len(boxes), dtype=int, device=device)
            })
        return images, targets

    @staticmethod
    def _collate_fn(batch):
        """Define a collate function to handle batches."""
        return tuple(zip(*batch))

    def train_dataloader(self):
        train_dataset = datasets.WIDERFace(
            root=self.root,
            split='train',
            download=False
        )
        return DataLoader(
            train_dataset, batch_size=4,
            shuffle=True, num_workers=8,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        test_dataset = datasets.WIDERFace(
            root=self.root,
            split='val',
            download=False
        )
        return DataLoader(
            test_dataset, batch_size=2,
            shuffle=False, num_workers=8,
            collate_fn=self._collate_fn
        )


def main():
    data = FacesData(root=r'/home/ff/myProject/KGT/myProjects/myDataset/widerface')
    model = FaceDetectionModel()

    save_callback = L.callbacks.ModelCheckpoint(
        monitor='val_loss',# TODO 监控的指标为平均绝对误差最小的,这一点和on_validation_epoch_end日志记录的指标是呼应的
        save_top_k=1,# TODO 这里的1，表示保存的模型中，只保存前4个最好结果模型权重文件
        mode='min',# TODO 表示保存当前误差最小的模型
        filename='{epoch}-{val_mae:.2f}'  # TODO 保存模型格式
    )
    trainer = L.Trainer(
        accelerator='gpu',
        devices=[1],
        max_epochs=100,
        precision='32',
        log_every_n_steps=10,
        callbacks=[save_callback,]
    )
    trainer.fit(model, data)

def loadModel():
    model = FaceDetectionModel()
    model.load_from_checkpoint(r'')
    return model

def predict():
    pass

if __name__ == "__main__":
    main()


