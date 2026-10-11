'''
# -*- encoding: utf-8 -*-
# 文件    : train.py
# 说明    : 对模型进行训练
# 时间    : 2022/06/27 16:58:41
# 作者    : Hito
# 版本    : 1.0
# 环境    : pytorch1.7
'''


import os
import torch
import shutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
from net.loss import BCELoss
from utils.utils import get_lr
from net.densebox import DenseBox
from utils.utils import mask_by_sel
from torch.utils.data import DataLoader
from dataset.dataloader import LPPatch_Offline


def loadDataset():
    # 加载数据集
    root_dir = r'/home/ff/myProject/KGT/myProjects/myDataset/patches/patches'
    train_set = LPPatch_Offline(root=root_dir, transform=None, size=(240, 240))
    batch_size = 10
    # 对数据集进行打包
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=8)
    print('load dataset done ...')
    return train_loader

def loadModel(device,resume = None):
    # network 加载网络模型
    net = DenseBox().to(device)
    # ---------------- whether to resume from checkpoint
    # 判断是否加载预训练模型
    if resume is not None:
        if os.path.isfile(resume):
            net.load_state_dict(torch.load(resume))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Note]: invalid resume path @ %s, resume failed.' % resume)

    print('load model done ...')
    return net

def train(num_epoch=30,lambda_loc=3.0,base_lr=1e-4,
          resume=None, save_folder='./weights'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = loadDataset()

    denseBox = loadModel(resume = resume,device = device)
    # ---------------- loss functions
    # element-wise L2 loss 定义损失函数
    loss_func = nn.MSELoss(reduce=False).to(device)
    # loss_func = nn.BCELoss(reduce=False).to(device)

    # optimization function
    # optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=9e-1, weight_decay=5e-4)  # 5e-4
    
    lr = 1e-3
    optimizer = optim.Adam(denseBox.parameters(),lr,weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,
                                                        eta_min=1e-5)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
    
    if os.path.exists(save_folder):
        #shutil.rmtree() 表示递归删除文件夹下的所有子文件夹和子文件。
        shutil.rmtree(save_folder)
    #新建指定的文件夹
    os.makedirs(save_folder)
    
    
    print('\nTraining...')
    denseBox.train()

    for epoch_i in range(num_epoch):
        for batch_i, (data, label_map, loss_mask) in enumerate(train_loader):
            # ------------- put data to device
            data, label_map = data.to(device), label_map.to(device)  # n,3,240,240   # n,5,60,60
            # ------------- forward pass
            # score_out, loc_out = net.forward(data)
            score_out, loc_out = denseBox(data)

            # ------------- loss calculation with hard negative mining
            score_map_gt = label_map[:, 0]  # n,60,60
            #TODO 获得score以及box
            score_map_gt = score_map_gt.unsqueeze(1)  # n,1,60,60
            loc_map_gt = label_map[:, 1:]   # n,4,60,60

            #TODO 得到非零位置的坐标
            positive_indices = torch.nonzero(score_map_gt)   # m ,4  m表示非0元素的个数，其实4就是每个元素的位置坐标
            #TODO 得到非零坐标的个数
            positive_num = positive_indices.size(0)

            # to keep the ratio of positive and negative sample to 1
            #TODO 得到正样本的占比
            negative_num = int(float(positive_num) / float(data.size(0)) + 0.5)
            #TODO 对得到的score求得概率
            score_out = torch.sigmoid(score_out)
            #TODO 求解score的损失值
            score_loss = loss_func(score_out, score_map_gt)
            # score_loss = BCELoss(score_out, score_map_gt)  # n,1,60,60

            # loc loss should be masked by label scores and to be summed
            #TODO 计算定位损失值
            loc_loss = loss_func(loc_out, loc_map_gt)  # n,4,60,60

            # TODO 难例挖掘 negative smapling... debug
            ones_mask = torch.ones([data.size(0), 1, 60, 60],
                                   dtype=torch.float32).to(device)
            #TODO 得到负样本的个数 = ones_mask - posi_mask
            neg_mask = ones_mask - score_map_gt
            #TODO 得到负样本的损失值
            negative_score_loss = score_loss * neg_mask

            #TODO 得到负样本的个数
            half_neg_num = int(negative_num * 0.5 + 0.5)
            negative_score_loss = negative_score_loss.view(data.size(0), -1)
            #TODO 得到负样本的前half_neg_num个值以及对应的索引值
            hard_negs, hard_neg_indices = torch.topk(input=negative_score_loss,
                                                     k=half_neg_num,
                                                     dim=1)

            rand_neg_indices = torch.zeros([data.size(0), half_neg_num],
                                           dtype=torch.long).to(device)
            #TODO 随机选择half_nge_num个负样本
            for i in range(data.size(0)):
                indices = np.random.choice(3600,  # 60 * 60
                                           half_neg_num,
                                           replace=False)
                indices = torch.Tensor(indices)
                rand_neg_indices[i] = indices

            #TODO  concatenate negative sample ids  所选定的负样本，包括损失最大的负样本和随机筛选的（各占负样本的一半），负样本总数为正样本的一半
            neg_indices = torch.cat((hard_neg_indices, rand_neg_indices), dim=1)

            neg_indices = neg_indices.cpu()
            positive_indices = positive_indices.cpu()

            # fill the loss mask
            mask_by_sel(loss_mask=loss_mask,
                        pos_indices=positive_indices,
                        neg_indices=neg_indices)

            # ------------- calculate final loss
            loss_mask = loss_mask.to(device)

            mask_score_loss = loss_mask * score_loss
            mask_loc_loss = loss_mask * score_map_gt * loc_loss

            loss = torch.sum(mask_score_loss)  + torch.sum(lambda_loc * mask_loc_loss)

            # ------------- back propagation
            # ------------- clear gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ------------- print loss
            iter_count = epoch_i * len(train_loader) + batch_i
            if iter_count % 10 == 0:
                print('=> epoch {} iter {:>3d}/{:>3d}'
                      ', total_iter {:>5d} '
                      ', lr {:>.8f} '
                      ', mask_loss {:>5.3f} '
                      ', local_loss {:>5.3f} '
                      '| loss {:>5.3f}'
                      .format(epoch_i + 1,
                              batch_i,
                              len(train_loader),
                              iter_count,
                              get_lr(optimizer),
                              torch.sum(mask_score_loss).item(),
                              torch.sum(lambda_loc * mask_loc_loss).item(),
                              loss.item()))

        # ------------ save checkpoint
        torch.save(denseBox.state_dict(), save_folder+'/model_' + str(epoch_i) + '.pth')
        print('<= {} saved.\n'.format(save_folder+'/model_' + str(epoch_i) + '.pth'))
        
        lr_scheduler.step()

    torch.save(denseBox.state_dict(), save_folder+'/model_final.pth')
    print('<= {} saved.\n'.format(save_folder+'/model_final.pth'))


if __name__ == "__main__":
    train(num_epoch=50, lambda_loc=3.0, resume=None)
