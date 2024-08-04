"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/8/1-22:04
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import os

import torch
import config
from PIL import Image
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
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
    target_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform,target_transform

class myDataset(Dataset):
    def __init__(self,dataset,imgs_transform):
        self.dataset = dataset
        self.imgs_transform = imgs_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        imgs_path,label = self.dataset[index]
        image = Image.open(imgs_path)
        image = self.imgs_transform(image)
        label = torch.tensor(label,dtype=torch.long)
        return image,label

def collete_fn(batch):
    imgs,labels = list(zip(*batch))
    tensor_imgs = []
    tensor_labels = []

    for i in range(len(imgs)):
        tensor_imgs.append(imgs[i])
        tensor_labels.append(labels[i])

    tensor_imgs = torch.stack(tensor_imgs)
    tensor_labels = torch.stack(tensor_labels)
    return tensor_imgs,tensor_labels

def loadDataset():
    transform , target_transform= loadTransform()
    #TODO 这里加载本地的数据集[daisy,dandelion,roses,sunflowers,tulips] == [黛西、蒲公英、玫瑰、向日葵、郁金香]
    dataset = ImageFolder(
        root=config.ROOT,
        transform=transform,
        target_transform=target_transform
    )
    class_to_index = dataset.class_to_idx
    print('class to index: {}'.format(class_to_index))
    total_size = dataset.__len__()
    print('total size: {}'.format(total_size))

    #TODO 划分训练集和验证集;其中step表示每个类别包含的图像数量
    step = int(total_size * (1 / len(class_to_index)))
    train_dataset = dataset.samples[:int(step * 0.8)]
    val_dataset = dataset.samples[int(step * 0.8):step]

    #TODO 从每一个类别中选择前80%作为训练集，20%作为验证集(注意这里的每一个子文件夹下面的图像数量不一定相同，但是都是取20%作为验证集)
    for i in range(step,total_size,step):
        train_dataset += dataset.samples[i:i + int(step * 0.8)]
        val_dataset += dataset.samples[i + int(step * 0.8):i + step]

    print('train size: {}'.format(train_dataset.__len__()))
    print('val size: {}'.format(val_dataset.__len__()))

    train_dataset = myDataset(train_dataset,transform)
    val_dataset = myDataset(val_dataset,transform)

    trainLoader = DataLoader(
        dataset=train_dataset,shuffle=True,
        batch_size=config.BATCH_SIZE,collate_fn=collete_fn
    )
    valLoader = DataLoader(
        dataset=val_dataset,shuffle=False,
        batch_size=1,collate_fn=collete_fn
    )
    return trainLoader,valLoader

if __name__ == '__main__':
    loadDataset()
    pass

