"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/7/28-20:18
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
# from dataset import loadDataset
from dataset_v2 import loadDataset

from model import Model
from model_finetune import modelFineTune
import config

def main():
    #TODO load dataset
    trainLoader,testLoader = loadDataset()

    #TODO load model
    # model = Model(in_channels=config.IN_CHANNELS,num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model = modelFineTune(num_classes=config.NUM_CLASSES,pretrained=config.PRETRAINED,
                          freeze_layers=config.FREEZE_LAYERS,isFreezeBackbone=config.ISFREEZEBACKBONE,
                          model_name='mobilenetv3').to(config.DEVICE)

    #TODO define optimizer and loss funcation
    loss_fn = nn.CrossEntropyLoss().to(config.DEVICE)
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=config.LR)

    #TODO load checkpoint model
    if config.RESUME:
        checkpoint = torch.load(config.RESUME,map_location='cpu')
        config.START_EPOCH = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        config.BEST_ACC = checkpoint['best_acc']
        print('load checkpoint is done ...')

    start_time = time.time()
    #TODO train phrase
    epoch_loss = []
    epoch_acc = []
    val_acc = []
    best_acc = 0
    for epoch in range(config.START_EPOCH,config.EPOCH):
        model.train()
        train_loss,train_acc = train_epoch(model,loss_fn,optimizer,trainLoader,epoch)
        if epoch % config.EVAL_EPOCH == 0:
            model.eval()
            acc = evaluate(model,testLoader,epoch)
            val_acc.append(acc)
            if acc > best_acc:
                best_acc = acc
                state_dict = {
                    'state_dict':model.state_dict(),
                    'optimizer':optimizer.state_dict,
                    'epoch':epoch,
                    'best_acc':best_acc
                }
                if config.ISFREEZEBACKBONE:
                    torch.save(
                        state_dict, os.path.join(config.OUTPUT, 'best_finetune_{}_{}.pth'.format(
                            round(best_acc, 3), epoch)))
                else:
                    torch.save(
                        state_dict,os.path.join(config.OUTPUT,'best_layers_{}_finetune_{}_{}.pth'.format(
                            config.FREEZE_LAYERS,round(best_acc,3),epoch)))
                # torch.save(state_dict,
                #            os.path.join(config.OUTPUT, 'best_p_0.5_nofinetune_{}_{}.pth'.format(round(best_acc, 3), epoch)))
        epoch_loss.append(train_loss)
        epoch_acc.append(train_acc)

    end_time = time.time()
    print('train time is: {}s'.format(end_time - start_time))
    plot(epoch_loss,mode = "{} finetune train loss".format(config.FREEZE_LAYERS))
    plot(epoch_acc,mode = "{} finetune train acc".format(config.FREEZE_LAYERS))
    plot(val_acc,mode = "{} finetune val acc".format(config.FREEZE_LAYERS))

    # plot(epoch_loss, mode="p_0.5 nofinetune train loss")
    # plot(epoch_acc, mode="p_0.5 nofinetune train acc")
    # plot(val_acc, mode="p_0.5 nofinetune val acc")

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
    # plt.show()
    plt.legend(labels = mode)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
    """
    p = 0.2 nofinetune train time is: 1859.5014691352844s
    p = 0.5 nofinetune train time is: 4755.051202535629s
    layer-5 finetune train time is: 3995.3461089134216s
    layer-10 finetune train time is: 2576.454952955246s
    finetune train time is: 753.2307982444763s
    train time is: 1913.6642224788666s
    """
    pass