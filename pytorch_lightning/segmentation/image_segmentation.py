"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/5/19-20:42
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import cv2
import time

import config
import numpy as np
from typing import Any
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import pytorch_lightning as L


class LitSegmentation(L.LightningModule):
    def __init__(self, lr = 0.001):
        super().__init__()
        self.lr = lr
        self.model = models.segmentation.fcn_resnet50(num_classes=21)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch,batch_idx):
        images, targets = batch
        outputs = self.model(images)['out']
        loss = self.loss_fn(outputs, targets.long().squeeze(1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)['out']
        loss = self.loss_fn(outputs, targets.long().squeeze(1))
        self.log("val_loss", loss)
        return loss

    def forward(self,x) -> Any:
        output = self.model(x)
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


class SegmentationData(L.LightningDataModule):
    def __init__(self,root):
        super().__init__()
        self.root = root

    # def prepare_data(self):
    #     datasets.VOCSegmentation(root="data", download=True)

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
        train_dataset = datasets.VOCSegmentation(
            root=self.root,
            transform=transform,
            year='2012',
            download=False,
            image_set="train",
            target_transform=target_transform
        )
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=4,
            shuffle=True, num_workers=8
        )

    def test_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
        test_dataset = datasets.VOCSegmentation(
            root=self.root,
            transform=transform,
            year='2012',
            download=False,
            image_set="test",
            target_transform=target_transform
        )
        return torch.utils.data.DataLoader(
            test_dataset, batch_size=2,
            shuffle=True, num_workers=8
        )

def main():
    model = LitSegmentation()
    data = SegmentationData(
        root=r'/home/ff/myProject/KGT/myProjects/myDataset/voc2012'
    )

    save_callback = L.callbacks.ModelCheckpoint(
        monitor='val_loss',  # TODO 监控的指标为平均绝对误差最小的,这一点和on_validation_epoch_end日志记录的指标是呼应的
        save_top_k=1,  # TODO 这里的1，表示保存的模型中，只保存前4个最好结果模型权重文件
        mode='min',  # TODO 表示保存当前误差最小的模型
        filename='{epoch}-{val_loss:.2f}'  # TODO 保存模型格式
    )

    trainer = L.Trainer(
        max_epochs=100,
        val_check_interval=5,
        log_every_n_steps=1,
        callbacks=[save_callback,]
    )
    trainer.fit(model, data)

def loadModel():
    model = LitSegmentation()
    model.load_from_checkpoint(
        checkpoint_path=r'/home/ff/myProject/KGT/myProjects/myProjects/PyTorchLighting/lightning_logs/version_0/checkpoints/epoch=9-step=999.ckpt'
    )
    return model

def segmentImage():
    model = loadModel()
    color_map = config.create_color_map(num_classes=len(config.palette))
    images_list = os.listdir(config.root)
    for imgName in images_list:
        starTime = time.time()
        img_path = os.path.join(config.root, imgName)
        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert("RGB")
        image = config.transform(img).unsqueeze(dim = 0).to(config.device)

        with torch.no_grad():
            output = model(image)

        out = output['out']
        aux = output['aux']
        endTime = time.time()

        print('out.shape: {}'.format(out.shape))
        print('aux.shape: {}'.format(aux.shape))

        prediction = out.squeeze(0).cpu().numpy()
        prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
        print(prediction)
        # TODO 可视化方式一 =================================================
        colorized_mask = config.save_images(image=img, mask=prediction, output_path=config.save_dir, image_file=imgName,
                                            palette=config.palette, num_classes=len(config.palette))
        # colorized_mask.show()
        #TODO # 可视化方式二 =================================================
        # color_image = config.output_to_color_image(prediction,color_map)
        # color_image = Image.fromarray(color_image)
        # color_image.save(os.path.join(config.save_dir,imgName))
        # color_image.show()

        print('segment {} time is {}s'.format(imgName,endTime - starTime))


def timeSegmentImage():
    model = loadModel()
    cap = cv2.VideoCapture(0)
    color_map = config.create_color_map(len(config.palette))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break

        img = cv2.resize(frame, dsize=(config.img_size, config.img_size))
        image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = config.transform(image).unsqueeze(dim=0).to(config.device)

        with torch.no_grad():
            output = model(image)

        out = output['out']
        aux = output['aux']

        print('out.shape: {}'.format(out.shape))
        print('aux.shape: {}'.format(aux.shape))

        prediction = out.squeeze(0).cpu().numpy()
        prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

        color_image = config.output_to_color_image(prediction, color_map)
        color_image = Image.fromarray(color_image)

        cv_img = np.array(color_image)
        cv_img = cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)

        cv2.imshow('img',cv_img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

import matplotlib.pyplot as plt

# 将张量转为 numpy 并显示
def show_sample(image, mask):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))  # [C, H, W] -> [H, W, C]
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='jet')
    plt.title("Mask")
    plt.show()

# image, mask = train_dataset[0]
# show_sample(image, mask)

if __name__ == "__main__":
   main()
   # segmentImage()
   pass