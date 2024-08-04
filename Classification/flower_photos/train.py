"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/7/28-20:18
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
import config
import os.path
from torch import nn
from tqdm import tqdm
from model import Model
import matplotlib.pyplot as plt
from dataset import loadDataset as datasetv1
from dataset_v2 import loadDataset as datasetv2

def main():
    #TODO 加载数据集
    trainLoader,testLoader = datasetv2()

    #TODO 加载模型
    model = Model(in_channel=config.IN_CHANNELS, num_classes=config.NUM_CLASSES).to(config.DEVICE)

    #TODO 定义优化器和损失函数
    loss_fn = nn.CrossEntropyLoss().to(config.DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LR)

    #TODO 加载中间保存的模型，继续上一步开始训练
    if config.RESUME:
        checkpoint = torch.load(config.RESUME, map_location='cpu')
        config.START_EPOCH = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        config.BEST_ACC = checkpoint['best_acc']
        print('load checkpoint is done ...')

    #TODO 开始训练
    epoch_loss = []
    epoch_acc = []
    best_acc = 0
    for epoch in range(config.START_EPOCH, config.EPOCH):
        model.train()
        train_loss,train_acc = train_epoch(model,optimizer,loss_fn,trainLoader,epoch)
        epoch_acc.append(train_acc)
        epoch_loss.append(train_loss)
        if epoch % config.EVAL_EPOCH  == 0:
            acc = evaluate(model,testLoader,epoch)
            if acc > best_acc:
                best_acc = acc
                state_dict = {
                    'state_dict':model.state_dict(),
                    'optimizer':optimizer.state_dict,
                    'epoch':epoch,
                    'best_acc':best_acc
                }
                torch.save(state_dict, os.path.join(config.OUTPUT, 'best_{}_.pth'.format(round(best_acc, 3))))

    plot(epoch_loss,mode='loss')
    plot(epoch_acc,mode='acc')
def train_epoch(model,optimizer,loss_fn,dataloader,epoch):
    total_loss = 0.
    total_acc = 0.
    total_step = 0
    loader = tqdm(dataloader,leave=True)
    loader.set_description('training ... ')
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
    loader.set_description('valing ... ')
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

def plot(logs,mode = 'loss'):
    plt.figure(figsize=(8,6))
    x = [i for i in range(len(logs))]
    plt.plot(x,logs,color = 'cyan')
    plt.savefig(os.path.join(config.OUTPUT, mode + '.png'))
    plt.show()

if __name__ == '__main__':
    main()
    pass