import os
import sys
os.environ['CUDA_VISIBLE_DEVICE']='1'

from prepare_data import (KittiDataset, VocDataset,
                          collater, Resizer,
                          AspectRatioBasedSampler,
                          Normalizer)
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import collections
import torch.optim as optim
from retinanet import model
import numpy as np
from tools import SplitKittiDataset
from retinanet import csv_eval
import config as cfg

from augmentation_zoo.MyGridMask import GridMask
from augmentation_zoo.SmallObjectAugmentation import SmallObjectAugmentation
from augmentation_zoo.Myautoaugment_utils import AutoAugmenter
from augmentation_zoo.RandomFlip import RandomFlip
from augmentation_zoo.Mixup_todo import mixup, mix_loss

"""
    author: zhenglin.zhou
    date: 20200724
"""

# os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_DEVICES

print('CUDA available: {}'.format(torch.cuda.is_available()))

def _make_transform():
    transform_list = list()
    if cfg.AUTOAUGMENT:
        transform_list.append(AutoAugmenter(cfg.AUTO_POLICY))
    if cfg.GRID:
        transform_list.append(GridMask(use_h=True, use_w=True,
                                       rotate=cfg.GRID_ROTATE,
                                       offset=cfg.GRID_OFFSET,
                                       ratio=cfg.GRID_RATIO,
                                       mode=cfg.GRID_MODE,
                                       prob=cfg.GRID_PROB))
    if cfg.RANDOM_FLIP:
        transform_list.append(RandomFlip())
    if cfg.SMALL_OBJECT_AUGMENTATION:
        transform_list.append(SmallObjectAugmentation(cfg.SOA_THRESH,
                                                      cfg.SOA_PROB,
                                                      cfg.SOA_COPY_TIMES,
                                                      cfg.SOA_EPOCHS,
                                                      cfg.SOA_ALL_OBJECTS,
                                                      cfg.SOA_ONE_OBJECT))
    transform_list.append(Normalizer())
    transform_list.append(Resizer())
    return transform_list

def _make_dataset():
    #TODO 目标只针对小目标增强
    transform_list = _make_transform()
    #TODO 默认为VOC数据集
    if cfg.DATASET_TYPE == 1:
        batch_size = cfg.VOC_BATCH_SIZE
        #TODO 加载VOC训练集和验证集
        dataset_train = VocDataset(cfg.VOC_ROOT_DIR, 'train',
                                   transform=transforms.Compose(transform_list))
        dataset_val = VocDataset(cfg.VOC_ROOT_DIR, 'val',
                                 transform=transforms.Compose(
                                     [Normalizer(), Resizer()]
                                 ))
    elif cfg.DATASET_TYPE == 2:
        root_dir = cfg.KITTI_ROOT_DIR
        batch_size = cfg.KITTI_BATCH_SIZE
        SplitKittiDataset(root_dir, 0.5)  # 分割KITTI数据集，50%训练集，50%测试集
        dataset_train = KittiDataset(root_dir, 'train', transform=transforms.Compose(transform_list))
        dataset_val = KittiDataset(root_dir, 'val', transform=transforms.Compose([Normalizer(), Resizer()]))
    return batch_size, dataset_train, dataset_val


def main():

    batch_size, dataset_train, dataset_val = _make_dataset()
    sampler = AspectRatioBasedSampler(dataset_train,
                                      batch_size=batch_size,
                                      drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=8,
                                  collate_fn=collater,
                                  batch_sampler=sampler)
    print("load dataset is done ...")

    retinanet = model.resnet18(num_classes=dataset_train.num_classes(),
                               pretrained=True)

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist =collections.deque(maxlen=500)

    retinanet.train()
    # retinanet.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    print('evaluating ...')
    average_precisions, mAP = csv_eval.evaluate(dataset_val, retinanet)
    print('average_precisions: {}   mAP: {} '.format(average_precisions,mAP))

    BEST_MAP = 0
    BEST_MAP_EPOCH = 0
    for epoch_num in range(cfg.EPOCHS):

        retinanet.train()
        # retinanet.freeze_bn()
        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            # try:
            optimizer.zero_grad()

            if cfg.MIXUP:
                data, lam = mixup(data)

            if torch.cuda.is_available():
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
            else:
                classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

            if cfg.MIXUP:
                classification_loss, regression_loss = mix_loss(classification_loss,
                                                                regression_loss,
                                                                lam)

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            if bool(loss == 0):
                continue
            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            optimizer.step()

            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))

            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            del classification_loss
            del regression_loss

        # except Exception as e:
        #     print(e)
        #     continue
        if epoch_num > 0 and epoch_num % cfg.eval_interval == 0:
            """ validation part """
            print('Evaluating dataset')
            average_precisions, mAP = csv_eval.evaluate(dataset_val, retinanet)
        if mAP > BEST_MAP:
            best_average_precisions = average_precisions
            BEST_MAP = mAP
            BEST_MAP_EPOCH = epoch_num
        scheduler.step(np.mean(epoch_loss))
        torch.save(retinanet.module, 'weights/{}_retinanet_{}.pt'.format('voc', epoch_num))
    retinanet.eval()

    print('\nBest_mAP:', BEST_MAP_EPOCH)
    for label in range(dataset_val.num_classes()):
        label_name = dataset_val.label_to_name(label)
        print('{}: {}'.format(label_name, best_average_precisions[label][0]))
    print('BEST MAP: ', BEST_MAP)
    # torch.save(retinanet, 'model_final.pt')

if __name__ == '__main__':

    main()


"""
small object augmentation + 90epochs
    Best_mAP: 90
    aeroplane: 0.7640969947388474
    bicycle: 0.6804498256943883
    bird: 0.6576336985990614
    boat: 0.40969361089370687
    bottle: 0.42201369093491653
    bus: 0.7166673687121604
    car: 0.6923749684198289
    cat: 0.8273532906357463
    chair: 0.37226816849511846
    cow: 0.5691019191304592
    diningtable: 0.3901193583400301
    dog: 0.7774691973994681
    horse: 0.6938712007336136
    motorbike: 0.7136954393566755
    person: 0.7316750100262398
    pottedplant: 0.30523341848151186
    sheep: 0.6540308197244709
    sofa: 0.503089907378012
    train: 0.7361053882980981
    tvmonitor: 0.6315532472188057
    BEST MAP:  0.6124248261605578

auto augmentation  + small object augmentation + 90epochs
    mAP:
    aeroplane: 0.8138397886192644
    bicycle: 0.6983633577904478
    bird: 0.7149951254340678
    boat: 0.49816360996000175
    bottle: 0.4240783184785889
    bus: 0.7208589475760898
    car: 0.6969562748095397
    cat: 0.8713283591629957
    chair: 0.389235917828056
    cow: 0.6417056611827153
    diningtable: 0.4018203566107144
    dog: 0.8176628807341757
    horse: 0.6915222595335763
    motorbike: 0.7205243561119066
    person: 0.7383907523885322
    pottedplant: 0.3152356564910122
    sheep: 0.678719871891284
    sofa: 0.5509777737969106
    train: 0.7794686509441838
    tvmonitor: 0.6100982423115957
    mAP:  0.6386973080827829
    
gridmask + small object augmentation + 90epochs
    mAP:
    aeroplane: 0.8046803295122336
    bicycle: 0.7443660530328308
    bird: 0.681367556492271
    boat: 0.461578003389003
    bottle: 0.42704128290451604
    bus: 0.7524736491573345
    car: 0.7253349155611932
    cat: 0.850653278596686
    chair: 0.39677413680060364
    cow: 0.5895224296140342
    diningtable: 0.44297711857755206
    dog: 0.8000457981294592
    horse: 0.7363172681019168
    motorbike: 0.7368746369108139
    person: 0.7723859922672991
    pottedplant: 0.34685732287801674
    sheep: 0.679235216834506
    sofa: 0.5403762798241876
    train: 0.7700818180052993
    tvmonitor: 0.6709662173600186
    mAP:  0.6464954651974887
"""