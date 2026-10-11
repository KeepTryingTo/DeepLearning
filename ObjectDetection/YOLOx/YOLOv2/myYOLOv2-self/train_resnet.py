"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/22 8:59
"""

"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/12 16:22
"""

import torch
from tqdm import tqdm
from configs.config import *
from dataset.mydataset import myDataset
from dataset.VOC import VOCDataSet
from utiles.loss import Yolov2Loss
from models.object.resnet50 import YOLOv2ResNet
from torch.utils.data import DataLoader
from utiles.scheduler import warmup_lr_scheduler
from utiles.misc import plot_map,plot_loss_and_lr

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#TODO load the dataset
def loadDataset():
    trainDataset = myDataset(
        rootDir=TRAIN_DIR_IMG,positionDir=TRAIN_DIR_LAB,
        S = S,B = B,num_classes=NUM_CLASSES,img_size=IMG_SIZE,transforms=transform
    )
    valDataset = myDataset(
        rootDir=VAL_DIR_IMG,positionDir=VAL_DIR_LAB,S = S,
        B = B,num_classes=NUM_CLASSES,img_size=IMG_SIZE,transforms=transform
    )
    trainLoader = DataLoader(
        dataset=trainDataset,batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,num_workers=NUM_WORKER,drop_last=False
    )
    valLoader = DataLoader(
        dataset=valDataset,batch_size=1,
        shuffle=SHUFFLE,num_workers=NUM_WORKER,drop_last=False
    )
    print('load the dataset done ...')
    return trainLoader,valLoader

def loadVOCDataset():
    trainDataset12 = VOCDataSet(
        voc_root=VOC_PATH,anchors=ANCHORS,year='2012',transforms=transform,
        train_set='trainval.txt',img_size=IMG_SIZE,
        S = S,B = B,num_classes=VOC_NUM_CLASSES,is_train=True
    )
    trainDataset07 = VOCDataSet(
        voc_root=VOC_PATH, anchors=ANCHORS, year='2007', transforms=transform,
        train_set='trainval.txt', img_size=IMG_SIZE,
        S=S, B=B, num_classes=VOC_NUM_CLASSES, is_train=True
    )
    valDataset = VOCDataSet(
        voc_root=VOC_PATH,anchors=ANCHORS,year='2007', transforms=transform,
        train_set='test.txt', img_size=IMG_SIZE,
        S=S, B=B, num_classes=VOC_NUM_CLASSES,is_train=False
    )
    trainLoader12 = DataLoader(
        dataset=trainDataset12,batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,shuffle=SHUFFLE,collate_fn=trainDataset12.collate_fn
    )
    trainLoader07 = DataLoader(
        dataset=trainDataset07, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER, shuffle=SHUFFLE, collate_fn=trainDataset07.collate_fn
    )
    valLoader = DataLoader(
        dataset=valDataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER, shuffle=SHUFFLE,collate_fn=valDataset.collate_fn
    )
    print('load voc dataset is done ...')
    return trainLoader07,trainLoader12,valLoader

#TODO load the model
def loadModel(pretrained = False,resume = None):
    model = YOLOv2ResNet(
        B = B,S = S,C = VOC_NUM_CLASSES
    )
    if pretrained:
        assert resume is not None,"if you set the pretrained is True,and the resume is not None"
        checkpoint = torch.load(resume,map_location='cpu')
        model.load_state_dict(checkpoint,strict=False)
    return model

def train_epoch(trainloader,model,optimizer,loss_fn,lr_scheduler,epoch):
    train_loss_epoch = 0
    train_loss = []
    for step, data in enumerate(trainloader):
        img, loc_targets, cls_targets, boxes_target = data
        img, loc_targets, cls_targets = img.to(DEVICE), loc_targets.to(DEVICE), cls_targets.to(DEVICE)
        boxes_target = [x.to(DEVICE) for x in boxes_target]
        #####################  forward ##########################
        output = model(img)
        loss = loss_fn(output, loc_targets, cls_targets, boxes_target)
        #####################  backward #########################
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        train_loss_epoch += loss.item()
        if step > 0:
            train_loss.append(train_loss_epoch / step)
        if step % STEP == 0 and step > 0:
            trainloader.set_postfix(
                epoch=epoch,
                step=step,
                loss=train_loss_epoch / step
            )

def main():
    # trainLoader,valLoader = loadDataset()
    trainLoader07, trainLoader12, valLoader = loadVOCDataset()
    model = loadModel(
        pretrained=False,
        resume=None
    ).to(DEVICE)

    #####################################################################
    # optimizer = torch.optim.Adam(
    #     params=model.parameters(),lr=BASE_LR,weight_decay=WEIGHT_DEACY
    # )
    lr_scheduler = None
    # lr_scheduler = warmup_lr_scheduler(
    #     optimizer=optimizer,
    #     warmup_iters=10,warmup_factor=0.9
    # )
    #################################################################################
    optimizer = torch.optim.SGD(
        model.parameters(), lr=BASE_LR,
        momentum=0.9, weight_decay=5e-4
    )

    pretrained = None
    start_epoch = 0
    if pretrained is not None:
        checkpoint = torch.load(pretrained, map_location='cpu')
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
    #################################################################################

    loss_fn = Yolov2Loss(
        img_size=IMG_SIZE, S=S, B=B, num_classes=VOC_NUM_CLASSES, anchors=ANCHORS, device=DEVICE,
        lambda_coord=5., lambda_noobj=.5, lambda_prior=5., lambda_obj=5., lambda_class=5., eps=EPSILON
    )
    best_miou = 0.
    min_val_loss = np.finfo(np.float32).max
    train_loss = []
    val_loss = []
    for epoch in range(start_epoch,EPOCHS):
        trainloader = tqdm(iterable=trainLoader12, desc="train is:", leave=True)
        model.train()
        train_epoch(trainloader, model, optimizer, loss_fn, lr_scheduler, epoch)
        trainloader = tqdm(iterable=trainLoader07, desc="train is:", leave=True)
        train_epoch(trainloader, model, optimizer, loss_fn, lr_scheduler, epoch)
        ################# evaluate #############################
        val_loss_epoch = 0
        model.eval()
        if epoch % VAL_STEP == 0:
            loader = tqdm(iterable=valLoader, desc="rs50 v:", leave=True)
            for step, data in enumerate(loader):
                img, loc_targets, cls_targets, boxes_target = data
                img, loc_targets, cls_targets = img.to(DEVICE), loc_targets.to(DEVICE), cls_targets.to(DEVICE)
                boxes_target = [x.to(DEVICE) for x in boxes_target]

                #####################  forward ##########################
                output = model(img)
                loss = loss_fn(output, loc_targets, cls_targets, boxes_target)
                val_loss_epoch += loss.item()
                if step > 0:
                    val_loss.append(val_loss_epoch / step)
                #####################  backward #########################
                if step % STEP == 0 and step > 0:
                    loader.set_postfix(
                        epoch=epoch,
                        step=step,
                        loss=loss.item() / step
                    )
            val_loss_epoch_avg = val_loss_epoch / len(valLoader)
            if min_val_loss > val_loss_epoch_avg:
                min_val_loss = val_loss_epoch_avg
                state = {
                    'model': model.state_dict(),
                    'loss':min_val_loss,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
                root_path = r'./weights'
                torch.save(state,f'{root_path}/{round(min_val_loss,3)}_resnet_losses_t_best_model.pth.tar')

    #plot the train loss and val loss
    plot_loss_and_lr(train_loss,val_loss)


if __name__ == '__main__':
    main()
    # print(np.finfo(np.float32).max)
    pass