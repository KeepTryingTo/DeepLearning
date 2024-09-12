"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/8/3-19:51
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset,random_split
import config


def loadTransform():
    transform = transforms.Compose([
        transforms.Resize(size=(config.IMG_SIZE,config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std = [0.229,0.224,0.225]
        )
    ])
    return transform

def loadDataset():
    transform = loadTransform()

    #TODO 加载本地的数据集
    dataset = ImageFolder(
        root=config.ROOT,
        transform=transform
    )
    class_to_index = dataset.class_to_idx
    print('class to index: {}'.format(class_to_index))
    total_size = dataset.__len__()
    print('total size: {}'.format(total_size))

    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.2)
    train_dataset,val_dataset = random_split(dataset,lengths = [train_size,val_size])

    trainLoader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=config.BATCH_SIZE)
    valLoader = DataLoader(dataset=val_dataset,shuffle=False,batch_size=1)

    return trainLoader,valLoader


if __name__ == '__main__':
    loadDataset()
    pass
