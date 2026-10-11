import os
import operator

import torch
import torch.utils.data
from torch.optim.lr_scheduler import StepLR

from engine.trainer import Trainer
from model.squeezedet import SqueezeDetWithLoss
from utils.config import Config
from utils.model import load_model, load_official_model, save_model
from utils.logger import Logger

from src.datasets.voc0712 import VOCDetection
from src.datasets.data_augment import preproc

from src.datasets.data_augment import BaseTransform


def loadVOCDataset():
    root_dir = r'/data1/KTG/myDataset/VOC'
    VOC = dict(
        train_sets=[('2007', 'trainval'), ('2012', 'trainval')],
        eval_sets=[('2007', 'test')],
    )
    img_size = (512,768)
    rgb_means = (103.94, 116.78, 123.68)
    p = 0.6
    _preproc = preproc(resize=img_size,
                       rgb_means=rgb_means,
                       p=p)
    trainDataset = VOCDetection(
        img_size = img_size,
        root=root_dir,image_sets=VOC['train_sets'],
        preproc=_preproc
    )

    valDataset = VOCDetection(
        img_size=img_size,
        root=root_dir, image_sets=VOC['eval_sets'],
        preproc=_preproc
    )

    return trainDataset, valDataset


def train(cfg):
    #TODO 加载数据集
    train_dataset, val_dataset = loadVOCDataset()

    #TODO 根据数据集的类型，更新参数信息，比如数据集包含的类别
    cfg = Config().update_dataset_info(cfg, train_dataset)
    Config().print(cfg)
    logger = Logger(cfg)

    model = SqueezeDetWithLoss(cfg)
    if cfg.load_model is not None:
        if (cfg.load_model.endswith('f364aa15.pth') or
                cfg.load_model.endswith('a815701f.pth')):
            model = load_official_model(model, cfg.load_model)
        else:
            model = load_model(model, cfg.load_model)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    lr_scheduler = StepLR(optimizer, 60, gamma=0.5)

    trainer = Trainer(model, optimizer, lr_scheduler, cfg)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               num_workers=cfg.num_workers,
                                               pin_memory=True,
                                               shuffle=True,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=cfg.num_workers,
                                             pin_memory=True,
                                             drop_last=True)

    metrics = trainer.metrics if cfg.no_eval else trainer.metrics + ['mAP']
    best = 1E9 if cfg.no_eval else 0
    better_than = operator.lt if cfg.no_eval else operator.gt

    for epoch in range(1, cfg.num_epochs + 1):
        train_stats = trainer.train_epoch(epoch, train_loader)
        logger.update(train_stats, phase='train', epoch=epoch)

        save_path = os.path.join(cfg.save_dir, 'model_last.pth')
        save_model(model, save_path, epoch)

        if epoch % cfg.save_intervals == 0:
            save_path = os.path.join(cfg.save_dir, 'model_{}.pth'.format(epoch))
            save_model(model, save_path, epoch)

        if cfg.val_intervals > 0 and epoch % cfg.val_intervals == 0:
            val_stats = trainer.val_epoch(epoch, val_loader)
            logger.update(val_stats, phase='val', epoch=epoch)

            loss = val_stats['loss']
            if better_than(loss, best):
                best = loss
                save_path = os.path.join(cfg.save_dir, f'voc_model_{str(round(best,3))}_best.pth')
                save_model(model, save_path, epoch)

        logger.plot(metrics)
        # logger.print_bests(metrics)

    torch.cuda.empty_cache()
