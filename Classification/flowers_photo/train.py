"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/7/28-20:18
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
# from dataset import loadDataset
from dataset_v2 import loadDataset
from torchvision import models,transforms

from model import Model
import config

def main():
    #TODO load dataset
    trainLoader,testLoader = loadDataset()

    #TODO load model
    model = Model(in_channels=config.IN_CHANNELS,num_classes=config.NUM_CLASSES).to(config.DEVICE)

    #TODO define optimizer and loss funcation
    loss_fn = nn.CrossEntropyLoss().to(config.DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=config.LR)

    #TODO load checkpoint model
    if config.RESUME:
        checkpoint = torch.load(config.RESUME,map_location='cpu')
        config.START_EPOCH = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        config.BEST_ACC = checkpoint['best_acc']
        print('load checkpoint is done ...')

    #TODO train phrase
    epoch_loss = []
    epoch_acc = []
    best_acc = 0
    for epoch in range(config.START_EPOCH,config.EPOCH):
        model.train()
        train_loss,train_acc = train_epoch(model,loss_fn,optimizer,trainLoader,epoch)
        if epoch % config.EVAL_EPOCH == 0:
            model.eval()
            acc = evaluate(model,testLoader,epoch)
            if acc > best_acc:
                best_acc = acc
                state_dict = {
                    'state_dict':model.state_dict(),
                    'optimizer':optimizer.state_dict,
                    'epoch':epoch,
                    'best_acc':best_acc
                }
                torch.save(state_dict,os.path.join(config.OUTPUT,'best_{}.pth'.format(round(best_acc,3))))
        epoch_loss.append(train_loss)
        epoch_acc.append(train_acc)


def train_epoch(model,loss_fn,optimizer,dataloader,epoch):
    total_loss = 0.
    total_acc = 0.
    total_step = 0
    loader = tqdm(dataloader,leave=True)
    loader.set_description('training...')
    for step,data in enumerate(loader):
        imgs,labels = data
        imgs = imgs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        output = model(imgs)
        loss = loss_fn(output,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_step += imgs.size()[0]
        prediction = torch.argmax(output,dim=-1)
        total_acc += (prediction == labels).sum().item()

        loader.set_postfix(
            epoch = epoch,
            step = step,
            loss = total_loss / total_step,
            acc = total_acc / total_step
        )

    return total_loss / total_step,total_acc / total_step

def evaluate(model,dataloader,epoch):
    total_acc = 0.
    total_step = 0
    loader = tqdm(dataloader, leave=True)
    loader.set_description('evaling...')
    for step, data in enumerate(loader):
        imgs, labels = data
        imgs = imgs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        output = model(imgs)

        total_step += imgs.size()[0]
        prediction = torch.argmax(output, dim=-1)
        total_acc += (prediction == labels).sum().item()

        loader.set_postfix(
            epoch=epoch,
            step=step,
            acc=total_acc / total_step
        )

    return total_acc / total_step

def plot(logs,mode = 'loss'):
    plt.figure(figsize=(8,6))
    x = [i for i in range(len(logs))]
    plt.plot(x,logs,color = 'cyan')
    plt.savefig(os.path.join(config.OUTPUT,mode + '.png'))
    plt.show()


if __name__ == '__main__':
    main()
    pass