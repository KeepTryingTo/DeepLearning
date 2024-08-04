"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/8/1-22:16
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import torch
import config
from PIL import Image
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision.datasets import ImageFolder

def loadTransform():
    #TODO 注意这里加载的图像是RGB图像，因此归一化的时候需要对三个通道分别对其进行归一化操作
    transform = transforms.Compose([
        transforms.Resize(size=(config.IMG_SIZE,config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])
    return transform

def loadDataset():
    transform = loadTransform()
    #TODO 这里加载本地的数据集[daisy,dandelion,roses,sunflowers,tulips] == [黛西、蒲公英、玫瑰、向日葵、郁金香]
    dataset = ImageFolder(
        root=config.ROOT,
        transform=transform
    )
    class_to_index = dataset.class_to_idx
    print('class to index: {}'.format(class_to_index))
    total_size = dataset.__len__()
    print('total size: {}'.format(total_size))

    #TODO 划分训练集和验证集
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.2)
    train_dataset,val_dataset = random_split(dataset,[train_size,val_size])

    trainLoader = DataLoader(
        dataset=train_dataset,shuffle=True,
        batch_size=config.BATCH_SIZE
    )
    valLoader = DataLoader(
        dataset=val_dataset,shuffle=False,
        batch_size=1
    )
    return trainLoader,valLoader

if __name__ == '__main__':
    loadDataset()
    pass

