'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''
import tqdm
import torch
import math,time
from torch import nn
from configs.config import *
from models.backbone import EfficientDetBackbone
from utiles.loss import FocalLoss
from utiles.plot import plot_loss
from utiles.encoder import Encoder
from torch.utils.data import DataLoader,Dataset
from models.efficientdet.voc import VOCDataset
# from models.efficientdet.dataset import *
from models.efficientdet.augmentations import *

class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss(device=DEVICE)
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression,
                                                anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression,
                                                anchors, annotations)
        return cls_loss, reg_loss

def loadVOCDataset():
    train_dataset_12=VOCDataset(
        root_dir=VOC_2012_PATH,transform = transforms.Compose(
            [
                Normalizer(mean=mean, std=std),
                Augmenter(),
                Resizer(input_sizes[compound_coef])
            ]
        ),
        is_training=True
    )
    train_dataset_07 = VOCDataset(
        root_dir=VOC_2007_PATH, transform=transforms.Compose(
            [
                Normalizer(mean=mean, std=std),
                Augmenter(),
                Resizer(input_sizes[compound_coef])
            ]
        ),
        is_training=True
    )
    val_dataset_07=VOCDataset(
        root_dir=VOC_2007_PATH,transform=transforms.Compose(
            [
                Normalizer(mean=mean, std=std),
                Resizer(input_sizes[compound_coef])
            ]
        ),
        is_training=False
    )
    train_loader_12 = torch.utils.data.DataLoader(
        train_dataset_12, batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collater
    )
    train_loader_07 = torch.utils.data.DataLoader(
        train_dataset_07, batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collater
    )
    val_loader_07 = torch.utils.data.DataLoader(
        val_dataset_07, batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collater
    )
    print('load the dataset is done ...')
    return train_loader_07,train_loader_12,val_loader_07,val_dataset_07

def loadModel(params,resum = None,pretrained = None):
    model = EfficientDetBackbone(
        num_classes=len(params.obj_list),
        compound_coef=0,
        ratios=eval(params.anchors_ratios),
        scales=eval(params.anchors_scales),
        load_weights=False
    ).to(DEVICE)
    start_epoch = 0
    optimizer = None
    #TODO 加载之前已经未完全训练完而保存的模型
    if resum is not None:
        checkpoint = torch.load(resum,map_location='cpu')
        optimizer = checkpoint['optimizer']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
    #TODO 加载GitHub中预训练在COCO上的模型
    if pretrained is not None:
        from collections import OrderedDict
        checkpoint = torch.load(pretrained,map_location='cpu')
        state_dict = OrderedDict()
        for key in list(checkpoint.keys()):
            if key.startswith('classifier'):
                continue
            else:
                state_dict[key] = checkpoint[key]
        model.load_state_dict(state_dict,strict=False)
        if head_only:
            def freeze_backbone(m):
                classname = m.__class__.__name__
                for ntl in ['EfficientNet', 'BiFPN']:
                    if ntl in classname:
                        for param in m.parameters():
                            param.requires_grad = False

            model.apply(freeze_backbone)
            print('[Info] freezed backbone')
    print('load the model is done ...')
    return model,start_epoch,optimizer

import yaml
class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def train_epoch(
        loss_fn,
        optimizer,
        train_loader,
        params
):
    loss_cls_epoch = 0
    loss_reg_epoch = 0

    loop = tqdm.tqdm(train_loader, leave=True)
    loop.set_description('training...')
    for epoch_step, data in enumerate(loop):
        batch_imgs, annots = data['img'], data['annot']
        batch_imgs = batch_imgs.to(DEVICE)
        annots = annots.to(DEVICE)

        optimizer.zero_grad()
        # model的参数默认为训练模式
        cls_loss, reg_loss = loss_fn(
            imgs=batch_imgs,
            annotations=annots,
            obj_list=params.obj_list
        )
        # 计算平均损失值
        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()

        loss = cls_loss + reg_loss
        loss.backward()
        optimizer.step()

        loss_cls_epoch += cls_loss.item()
        loss_reg_epoch += reg_loss.item()

        if epoch_step % STEP == 0 and epoch_step > 0:
            loop.set_postfix(
                loss_cls=loss_cls_epoch / epoch_step,
                loss_reg=loss_reg_epoch / epoch_step,
            )

    return loss_reg_epoch / len(train_loader), loss_cls_epoch / len(train_loader)

def val_epoc(
        loss_fn,
        val_loader_07,
        params
):
    loss_cls_epoch = 0
    loss_reg_epoch = 0

    loop = tqdm.tqdm(val_loader_07, leave=True)
    loop.set_description('valing...')
    for epoch_step, data in enumerate(loop):
        batch_imgs, annots = data['img'], data['annot']
        batch_imgs = batch_imgs.to(DEVICE)
        annots = annots.to(DEVICE)
        # model的参数默认为训练模式
        cls_loss, reg_loss = loss_fn(
            imgs=batch_imgs,
            annotations=annots,
            obj_list=params.obj_list
        )
        # 计算平均损失值
        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()

        loss_cls_epoch += cls_loss.item()
        loss_reg_epoch += reg_loss.item()

        if epoch_step % STEP == 0 and epoch_step > 0:
            loop.set_postfix(
                loss_cls=loss_cls_epoch / epoch_step,
                loss_reg=loss_reg_epoch / epoch_step
            )

    return loss_cls_epoch / len(val_loader_07), loss_reg_epoch / len(val_loader_07)

def train():
    params = Params(f'configs/voc.yml')
    model,start_epoch,optim = loadModel(params,resum=None,
                                        pretrained=None)
    train_loader_07,train_loader_12,val_loader_07,val_dataset_07 = loadVOCDataset()
    loss_fn = ModelWithLoss(model)
    optimizer = torch.optim.SGD(params=model.parameters(),lr=BASE_LR)
    if optim is not None:
        optimizer.load_state_dict(optim)
    min_loss_save = torch.inf
    # val phrase
    v_total_loss = 0.
    v_loss_reg = []
    v_loss_cls = []
    # train phrase
    t_total_loss = 0.
    t_loss_reg = []
    t_loss_cls = []
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t_loss_reg_e, t_loss_cls_e = train_epoch(loss_fn,optimizer,train_loader_07,params)
        t_loss_reg.append(t_loss_reg_e)
        t_loss_cls.append(t_loss_cls_e)
        t_loss_reg_e, t_loss_cls_e = train_epoch(loss_fn,optimizer,train_loader_12,params)
        t_loss_reg.append(t_loss_reg_e)
        t_loss_cls.append(t_loss_cls_e)

        with torch.no_grad():
            if epoch % VAL_STEP == 0:
                model.eval()
                v_loss_reg_e, v_loss_cls_e = val_epoc(loss_fn,val_loader_07,params)
                v_loss_reg.append(v_loss_reg_e)
                v_loss_cls.append(v_loss_cls_e)
                if v_loss_reg_e + v_loss_cls_e < min_loss_save:
                    min_loss_save = v_loss_reg_e + v_loss_cls_e
                    state = {
                        'model':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'epoch':epoch,
                        'min_loss':min_loss_save
                    }
                    torch.save(state,f'weights/voc_{epoch}_{round(min_loss_save,3)}_efficientDet_D0.pth.tar')
    plot_loss(loss_cls=t_loss_cls,loss_reg=t_loss_reg)
    plot_loss(loss_cls=v_loss_cls,loss_reg=v_loss_reg)

if __name__ == '__main__':
    train()
    pass







