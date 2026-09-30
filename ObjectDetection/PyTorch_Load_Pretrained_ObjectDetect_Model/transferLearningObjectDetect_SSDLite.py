"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/2/1 14:16
"""
import os
import cv2
import time
import torch
import torchvision
from torch import nn
from readData import *
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

#修改模型分类头的类别数
def modelSSD320(num_classes):
    #加载在COCO数据集上训练的预训练模型
    modelSSDLite=ssdlite320_mobilenet_v3_large(pretrained=True,progress=True)

    # replace the classifier with a new one, that has
    # 将分类器替换为具有用户定义的 num_classes的新分类器
    # 获取分类器的输入参数的数量
    # c_in_features=[modelSSDLite.head.classification_head.module_list[i].in_channels for i in range(len(modelSSDLite.head.classification_head.module_list))]
    c_in_features=[]
    norm_Layers=[]
    for i in range(len(modelSSDLite.head.classification_head.module_list)):
        in_channels_1=modelSSDLite.head.classification_head.module_list[i][0][0].in_channels
        normLayer=modelSSDLite.head.classification_head.module_list[i][0][1]
        c_in_features.append(in_channels_1)
        norm_Layers.append(normLayer)

    num_anchors=modelSSDLite.anchor_generator.num_anchors_per_location()
    # # 用新的头部替换预先训练好的头部
    modelSSDLite.head.classification_head=SSDLiteClassificationHead(in_channels=c_in_features,num_anchors=num_anchors,
                                                                    num_classes=num_classes,norm_layer=torch.nn.BatchNorm2d)

    return modelSSDLite


def train():
    # 训练数据类别
    classes = ['__background__','person']

    # 从文件中导入训练数据集
    trainDataset = myDataset_1(rootDir='../data/ssdLite320_mobilenet_v3_large/train_img',
                               positionDir='../data/ssdLite320_mobilenet_v3_large/train_txt')
    testDataset=myDataset_1(rootDir='../data/ssdLite320_mobilenet_v3_large/test_img',
                            positionDir='../data/ssdLite320_mobilenet_v3_large/test_txt')

    # 加载数据集
    trainDataLoader = DataLoader(dataset=trainDataset, batch_size=16, shuffle=True,
                                 num_workers=0, drop_last=True)
    testDataLoader = DataLoader(dataset=testDataset, batch_size=16, shuffle=True,
                                 num_workers=0, drop_last=False)
    print('-----------------------------------------加载数据完成-------------------------------------------------')

    # 导入训练模型
    # myModel=myBuildModel_2()
    device='cpu'
    myModel = modelSSD320(num_classes=len(classes))
    # print(myModel)
    myModel=myModel.to(device)
    #查看模型参数
    # print('modelParameters: {}'.format(stat(myModel, (3, 224, 224))))

    # 获取模型所有可训练参数
    params = [p for p in myModel.parameters() if p.requires_grad]
    # 定义优化器
    learning_rate = 0.001
    # 如果这里的优化器采用SGD的话，可能准确率很难提高(注意这里的可优化的参数是未冻结的层)
    optimizer = torch.optim.SGD(params=params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    # 学习速率衰减，这些具体的浮点数参数参照官网进行设定的
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)

    # 迭代的次数
    epoches = 20

    # 将损失值和准确率保存起来，用于最后画出图形
    PlotTestCorrect = []
    PlotTrainLoss = []

    # 定义写日志文件
    writer = SummaryWriter(log_dir='logs/' + str(time.strftime('%Y_%m_%d_%H_%M_%S')))

    # 开始训练
    for epoch in range(epoches):
        # 训练的次数
        trainStep = 0
        # 训练集上的损失
        trainLoss = 0
        traintotal = 0
        print('--------------------------------------第{}轮的训练------------------------------------'.format(epoch + 1))
        # 每一次迭代的训练
        myModel.train()
        for dataset in trainDataLoader:
            imgs, positions, labels = dataset
            imgs, positions, labels = imgs.to(device),positions.to(device),labels.to(device)
            images = list(image for image in imgs)
            targets = []

            # print('imgs.type: {}'.format(type(imgs)))
            # print('imgs.shape: {}'.format(imgs.shape))
            # print('position.shape: {}'.format(positions.shape))
            # print('labels.shape: {}'.format(labels.shape))
            # 将坐标和对应的标签存放在一个字典当中
            for i in range(len(images)):
                d = {}
                #由于这里是单个目标检测，所以需要对其维度进行变换一下[4]=>[1,4]
                d['boxes'] = positions[i]
                # print('positions[i].shape: {}'.format(positions[i].shape))
                d['labels'] = labels[i]
                # print('labels[i].shape: {}'.format(labels[i].shape))
                targets.append(d)

            # print('images.shape: {}'.format(len(images)))
            # print('targets.shape: {}'.format(len(targets)))
            # print('targets: {}'.format(targets))

            # 输入网路
            output = myModel(images,targets)
            # print(output)
            # print('output: {}'.format(output))
            loss_classifier=output['classification']
            loss_bbox=output['bbox_regression']

            # 将损失值相加，进行全局优化
            loss = loss_bbox+loss_classifier
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            # loss.requires_grad_(True)
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 学习速率衰步数记录
            lr_scheduler.step()
            # 步数加一
            trainStep = trainStep + 1
            with torch.no_grad():
                traintotal += imgs.shape[0]
                # 计算总的损失
                trainLoss += loss.item()
                if trainStep % 50 == 0:
                    print('-----------------step: {}--------------loss: {:.6f}----------------'.format(trainStep,
                                                                                                   trainLoss * 1.0 / traintotal
                                                                                                   ))
        writer.add_scalar(tag='trainLoss', scalar_value=trainLoss * 1.0 / traintotal, global_step=epoch + 1)

        testtotal=0
        testCorrect=0
        myModel.eval()
        with torch.no_grad():
            for dataset in testDataLoader:
                imgs, positions, labels = dataset
                imgs, positions, labels = imgs.to(device), positions.to(device), labels.to(device)
                images = list(image for image in imgs)
                # 输入网路
                output = myModel(images)
                confidence = 0
                for i in range(len(images)):
                    confidence+=output[i]['scores'][0].item()
                testtotal += imgs.shape[0]
                # 计算总的损失
                testCorrect += confidence
            print('----------------------------------------testCorrect: {:.6f}----------------'.format(testCorrect * 1.0 / testtotal
                                                                                                       ))
        writer.add_scalar(tag='testCorrect', scalar_value=testCorrect * 1.0 / testtotal, global_step=epoch + 1)
        PlotTrainLoss.append(trainLoss * 1.0 / traintotal)
        PlotTestCorrect.append(testCorrect*1.0/testtotal)
    writer.close()
    # 保运模型
    torch.save(myModel, 'myModelBestssd320_1.pth')
    # 使用matplotlib绘制图形
    plt.plot(range(1, epoches + 1), PlotTrainLoss, label='trainLoss')
    plt.legend()
    plt.show()
    plt.plot(range(1, epoches + 1), PlotTestCorrect, label='testCorrect')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    train()
    # modelSSD320(2)