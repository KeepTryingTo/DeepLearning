"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/24 13:37
"""


import torch
from tqdm import tqdm
from configs.config import *
from torchvision import models
from utiles.losses import Yolov1Loss
from dataset.mydataset import myDataset
from dataset.VOC import VOCDataSet
from dataset.coco import build
from models.object.mydarknet import DarkNet
from models.object.mySelfModel import EDANet
from models.object.resnet50 import YOLOv1ResNet
from models.object.resnet_yolo import resnet50
from models.object.vgg_yolo import vgg16_bn
from torch.utils.data import DataLoader
from utiles.scheduler import warmup_lr_scheduler
from utiles.misc import plot_map,plot_loss_and_lr


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
        voc_root=VOC_PATH,year='2012',transforms=transform,
        train_set='trainval.txt',img_size=IMG_SIZE,
        S = S,B = B,num_classes=VOC_NUM_CLASSES
    )
    trainDataset07 = VOCDataSet(
        voc_root=VOC_PATH, year='2007', transforms=transform,
        train_set='trainval.txt', img_size=IMG_SIZE,
        S=S, B=B, num_classes=VOC_NUM_CLASSES
    )
    valDataset = VOCDataSet(
        voc_root=VOC_PATH,year='2007', transforms=transform,
        train_set='test.txt', img_size=IMG_SIZE,
        S=S, B=B, num_classes=VOC_NUM_CLASSES
    )
    trainLoader12 = DataLoader(
        dataset=trainDataset12,batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,shuffle=SHUFFLE
    )
    trainLoader07 = DataLoader(
        dataset=trainDataset07, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER, shuffle=SHUFFLE
    )
    valLoader = DataLoader(
        dataset=valDataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER, shuffle=SHUFFLE
    )
    print('load voc dataset is done ...')
    return trainLoader12,trainLoader07,valLoader


def loadCOCODataset():
    # 加载数据集
    dataset_train = build(
        image_set='train', coco_path=COCO_PATH,transforms=transform,
        COCO_LABELS_name_2_id_MAP=COCO_LABELS_name_2_id_MAP,
        COCO_LABELS_id_2_name_MAP = COCO_LABELS_id_2_name_MAP
    )
    dataset_val = build(
        image_set='val', coco_path=COCO_PATH,transforms=transform,
        COCO_LABELS_name_2_id_MAP=COCO_LABELS_name_2_id_MAP,
        COCO_LABELS_id_2_name_MAP=COCO_LABELS_id_2_name_MAP
    )
    trainLoader = DataLoader(
        dataset=dataset_train, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER, shuffle=SHUFFLE
    )
    valLoader = DataLoader(
        dataset=dataset_val, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER, shuffle=SHUFFLE
    )
    print('load voc dataset is done ...')
    return trainLoader, valLoader

#TODO load the model
def loadModel(pretrained = False,resume = None):
    model = vgg16_bn(num_classes = COCO_NUM_CLASSES)
    vgg = models.vgg16_bn(pretrained=True)
    new_state_dict = vgg.state_dict()
    dd = model.state_dict()
    for k in new_state_dict.keys():
        print(k)
        if k in dd.keys() and k.startswith('features'):
            print('yes')
            dd[k] = new_state_dict[k]
    model.load_state_dict(dd)
    if pretrained:
        assert resume is not None,"if you set the pretrained is True,and the resume is not None"
        checkpoint = torch.load(resume,map_location='cpu')
        model.load_state_dict(checkpoint,strict=False)
    return model


def train_epoch(trainloader,optimizer,model,loss_fn,lr_scheduler,epoch):
    train_loss_epoch = 0
    train_loss = []
    for step, data in enumerate(trainloader):
        img, gt_map = data
        img, gt_map = img.to(DEVICE), gt_map.to(DEVICE)

        #####################  forward ##########################
        output = model(img)
        loss = loss_fn(output, gt_map)
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
    # trainLoader,valLoader = loadVOCDataset()
    trainLoader12,trainLoader07,valLoader = loadVOCDataset()
    model = loadModel(
        pretrained=False
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

    loss_fn = Yolov1Loss(
        lambda_noobj=lambda_noobj,lambda_coord=lambda_coord,
        num_classes=VOC_NUM_CLASSES,B = B,eps = EPSILON
    )
    from utiles.losses_t import yoloLoss
    # loss_fn = yoloLoss(S = S,B = B,l_coord=5,l_noobj=0.5,num_classes=NUM_CLASSES)
    best_miou = 0.
    min_val_loss = np.finfo(np.float32).max
    train_loss = []
    val_loss = []
    for epoch in range(start_epoch,EPOCHS):
        trainloader12 = tqdm(iterable=trainLoader12, desc="rest is:", leave=True)
        val_loss_epoch = 0
        model.train()
        train_epoch(trainloader12, optimizer, model, loss_fn, lr_scheduler, epoch)
        trainloader07 = tqdm(iterable=trainLoader07, desc="rest is:", leave=True)
        train_epoch(trainloader07, optimizer, model, loss_fn, lr_scheduler, epoch)
        ################# evaluate #############################
        model.eval()
        if epoch % VAL_STEP == 0:
            loader = tqdm(iterable=valLoader, desc="rs50 val is:", leave=True)
            for step, data in enumerate(loader):
                img, gt_map = data
                img, gt_map = img.to(DEVICE), gt_map.to(DEVICE)

                #####################  forward ##########################
                output = model(img)
                loss = loss_fn(output, gt_map)
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
                torch.save(state,f'weights/{round(min_val_loss,3)}_vgg_losses_t_best_model.pth.tar')

    #plot the train loss and val loss
    plot_loss_and_lr(train_loss,val_loss)


if __name__ == '__main__':
    main()
    # print(np.finfo(np.float32).max)
    pass