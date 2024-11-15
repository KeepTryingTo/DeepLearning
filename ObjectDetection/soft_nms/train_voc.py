'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''
import tqdm
import torch
from configs.config import *
from models.dssd_detector import DSSDDetector,createCfg
from utiles.scheduler import WarmupMultiStepLR
from utiles.plot import plot_loss
from dataset.datasets.voc import VOCDataset,BatchCollator
from torch.utils.data import DataLoader
from dataset.build_transforms import build_transforms,build_target_transform

from voc_eval import begin


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def loadVOCDataset(cfg):
    # TODO 构建训练集的数据增强transforms
    train_transform = build_transforms(cfg, is_train=True)
    train_target_transform = build_target_transform(cfg) if True else None
    val_transform = build_transforms(cfg, is_train=False)
    val_target_transform = build_target_transform(cfg) if False else None

    train_dataset_12 = VOCDataset(
        data_dir=VOC_2012_PATH,split='trainval',
        transform=train_transform,
        target_transform=train_target_transform
    )
    train_dataset_07 = VOCDataset(
        data_dir=VOC_2007_PATH, split='trainval',
        transform=train_transform,
        target_transform=train_target_transform
    )
    val_dataset = VOCDataset(
        data_dir=VOC_2007_PATH, split='test',
        transform=val_transform,
        target_transform=val_target_transform
    )

    train_loader_12 = DataLoader(
        dataset=train_dataset_12,batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=True,num_workers=NUM_WORKER,
        collate_fn=BatchCollator(True)
    )
    train_loader_07 = DataLoader(
        dataset=train_dataset_07, batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKER,
        collate_fn=BatchCollator(True)
    )
    val_loader = DataLoader(
        dataset = val_dataset,batch_size=1,
        shuffle=False,num_workers=NUM_WORKER,
        collate_fn=BatchCollator(True)
    )
    class_names = val_dataset.class_names
    print('load the dataset is done ...')
    print('train dataset size (VOC2012.trainval + VOC2007.trainval): {}'.format(
        train_dataset_07.__len__() + train_dataset_12.__len__()))
    print('test dataset size : {}'.format(val_dataset.__len__()))

    return (train_loader_12,train_loader_07,val_loader,
            train_dataset_12,train_dataset_07,val_dataset,class_names)

def loadModel(cfg,resum = None):
    model = DSSDDetector(cfg=cfg)
    start_epoch = 0
    optimizer = None
    #加载之前已经未完全训练完而保存的模型
    if resum is not None:
        checkpoint = torch.load(resum,map_location='cpu')
        optimizer = checkpoint['optimizer']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
    print('load the model is done ...')
    return model,start_epoch,optimizer

def defineOptimzer(cfg,model,optim):
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.LR,
                                momentum=cfg.SOLVER.MOMENTUM,
                                weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    if optim is not None:
        optimizer.load_state_dict(optim)
    return optimizer

def do_epoch_train(model,loop,optimizer,scheduler,device):
    t_cls_loss = 0
    t_loc_loss = 0
    t_total_loss = 0
    for epoch_step, data in enumerate(loop):
        batch_imgs, batch_boxes_label, index = data
        batch_imgs = batch_imgs.to(device)
        batch_boxes_label = batch_boxes_label.to(device)

        if epoch_step > 0:
            scheduler.step()

        optimizer.zero_grad()
        # model的参数默认为训练模式
        loss_dict = model(batch_imgs, batch_boxes_label)
        reg_loss = loss_dict['reg_loss']
        cls_loss = loss_dict['cls_loss']

        loss = reg_loss + cls_loss
        loss.backward()
        optimizer.step()

        t_cls_loss += cls_loss.item()
        t_loc_loss += reg_loss.item()
        t_total_loss += loss.item()

        if epoch_step % STEP == 0 and epoch_step > 0:
            loop.set_postfix(
                closs=t_cls_loss / epoch_step,
                rloss=t_loc_loss / epoch_step,
                all_loss=t_total_loss / epoch_step,
            )

    return t_cls_loss, t_loc_loss, t_total_loss

def make_lr_scheduler(cfg, optimizer, milestones=None):
    return WarmupMultiStepLR(optimizer=optimizer,
                             milestones=cfg.SOLVER.LR_STEPS if milestones is None else milestones,
                             gamma=cfg.SOLVER.GAMMA,
                             warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                             warmup_iters=cfg.SOLVER.WARMUP_ITERS)

def train():
    cfg = createCfg(config_file=r'configs/resnet101_dssd320_voc0712.yaml')
    device = cfg.MODEL.DEVICE
    #TODO 加载模型
    model,start_epoch,optim = loadModel(cfg=cfg,resum=None)
    model.to(device)
    optimizer = defineOptimzer(cfg,model,optim=optim)
    # anchor & loss
    (train_loader_12, train_loader_07, val_loader,
     train_dataset_12, train_dataset_07, val_dataset,
     class_names) = loadVOCDataset(cfg)

    # LR_STEPS: [80000, 100000]
    num_gpu = 1
    milestones = [step // num_gpu for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    t_loss = []
    v_loss = []
    cls_file = r'configs/voc_classes.txt'

    for epoch in range(start_epoch,EPOCHS):

        model.train()
        #TODO 首先训练12年数据集
        loop = tqdm.tqdm(train_loader_12,leave=True)
        loop.set_description('training VOC12: [{}]'.format(epoch))

        t_cls_loss, t_loc_loss, t_total_loss_12 = do_epoch_train(model,
                                                              loop,
                                                              optimizer,
                                                              scheduler,
                                                              device)
        # TODO 其次训练07年数据集
        loop = tqdm.tqdm(train_loader_07, leave=True)
        loop.set_description('training VOC07: [{}]'.format(epoch))
        t_cls_loss, t_loc_loss, t_total_loss_07 = do_epoch_train(model,
                                                                 loop,
                                                                 optimizer,
                                                                 scheduler,
                                                                 device)

        t_loss.append(
            (t_total_loss_12 + t_total_loss_07) / (train_loader_12.__len__() +
                                                   train_loader_07.__len__())
        )

        #TODO 测试07数据集
        if epoch % VAL_EPOCHS == 0:
            model.eval()
            begin(cfg = cfg,model=model,
                  cls_file = cls_file,
                  eval_dataset=val_dataset,
                  device=device)
            state = {
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':epoch
            }
            torch.save(state,f'weights/voc_{epoch}_dssd.pth.tar')
    plot_loss(loss_train=t_loss,loss_val=v_loss)

if __name__ == '__main__':
    train()
    pass







