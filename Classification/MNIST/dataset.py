"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/7/28-20:19
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

def loadTransforms():
    #TODO MNIST数据集是单通道的，所以在进行归一化的时候需要注意
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])
    return train_transform,test_transform

def loadDataset():
    train_transform, test_transform = loadTransforms()

    train_dataset = datasets.MNIST(root='./dataset',train=True,download=True,transform=train_transform)
    test_dataset = datasets.MNIST(root='./dataset',train=False,download=True,transform=test_transform)

    print('train dataset size: {}'.format(len(train_dataset)))
    print('test dataset size: {}'.format(len(test_dataset)))

    trainLoader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True
    )
    testLoader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False
    )
    return trainLoader,testLoader

if __name__ == '__main__':
    loadDataset()


