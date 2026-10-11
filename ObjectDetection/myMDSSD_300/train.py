from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import (VOCroot, COCOroot, VOC_300,
                  VOC_512, COCO_300,
                  COCO_512, AnnotationTransform,
                  VOCDetection, detection_collate,
                  BaseTransform, preproc)
from data.coco import COCODetection
from layers.modules import MultiBoxLoss
from layers.functions import PriorBox
import time

from collections import OrderedDict


parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='mdssd',
                    help='fssd or mdssd version.')
parser.add_argument('-s', '--size', default='300',help='300 or 512 input size.')
parser.add_argument('-d', '--dataset',
                    default='COCO',
                    help='VOC or COCO dataset')
parser.add_argument('--basenet', default='./weights/vgg16_reducedfc.pth',
                help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=4,type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8,type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument(
    '--resume_net',
    default=r'runs/mdssd_COCO_epoches_50.pth',
    help='resume net for retraining')
parser.add_argument('--resume_epoch', default=50,type=int, help='resume iter for retraining')
parser.add_argument('-max','--max_epoch', default=300,type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='./runs/',help='Location to save checkpoint models')
args = parser.parse_args()



if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    train_sets = [('2014', 'train'),('2014', 'val')]
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'myFeature_FSSD_Vgg':
    from models.FSSD_vgg import build_net
elif args.version == 'mdssd':
    from models.mdssd import MDSSD300
else:
    print('Unkown version!')

#TODO 图像的大小，RGB归一化值
img_dim = (300,512)[args.size=='512']
rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'fssd']
p = (0.6,0.2)[args.version == 'fssd']
#TODO 类别数，batch大小
num_classes = (21, 81)[args.dataset == 'COCO']
batch_size = args.batch_size
#TODO 权重衰减率
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9

# net = build_net( img_dim, num_classes)
net = MDSSD300(num_classes=num_classes)

if args.resume_net == None and args.version == 'fssd':
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    net.base.load_state_dict(base_weights)

    def xavier(param):
        init.xavier_uniform(param)
    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    print('Initializing weights...')
    # TODO initialize newly added layers' weights with kaiming_normal method
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    if args.version == 'RFB_E_vgg':
        net.reduce.apply(weights_init)
        net.up_reduce.apply(weights_init)

if args.resume_net == None and args.version == 'mdssd':
    base_weights = torch.load(args.basenet)
    state_dict = OrderedDict()
    for key, value in net.base.state_dict().items():
        if key in base_weights.keys():
            state_dict[key] = base_weights[key]
            # print('load key :{}'.format(key))
    print('Loading base network...')
    net.base.load_state_dict(state_dict)

else:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.cuda:
    net.cuda()
    cudnn.benchmark = True


optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

criterion = MultiBoxLoss(num_classes, overlap_thresh=0.5,
                         prior_for_matching=True, bkg_label=0,
                         neg_mining=True, neg_pos=3,
                         neg_overlap=0.5, encode_target=False)
#TODO 生成anchor box
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()



def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    if args.dataset == 'VOC':
        dataset = VOCDetection(
            VOCroot, train_sets,
            preproc(
                resize=img_dim,
                rgb_means=rgb_means,
                p = p
            ), AnnotationTransform()
        )
        print('load VOC dataset ...')
    elif args.dataset == 'COCO':
        dataset = COCODetection(
            COCOroot,
            train_sets,
            preproc(
                img_dim,
                rgb_means,
                p
            )
        )
        print('load COCO dataset ...')
    else:
        print('Only VOC and COCO are supported now!')
        return

    #TODO 得到每一个epochs的训练步数，步数乘以total_epochs == total_iterations
    epoch_size = len(dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues = (stepvalues_VOC,stepvalues_COCO)[args.dataset=='COCO']
    print('Training',args.version, 'on', dataset.name)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
        checkpoint = torch.load(args.resume_net,map_location='cpu')
        net.load_state_dict(checkpoint)
    else:
        start_iter = 0

    lr = args.lr
    for iteration in range(start_iter, max_iter):
        #TODO 如果数据集迭代完，继续打包数据集，准备下一次的训练
        if iteration % epoch_size == 0:
            # TODO create batch iterator
            batch_iterator = iter(
                data.DataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                collate_fn=detection_collate)
            )
            loc_loss = 0
            conf_loss = 0
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 ==0 and epoch > 200):
                torch.save(net.state_dict(),
                           args.save_folder+args.version+'_'+args.dataset + '_epoches_'+
                           repr(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        #TODO 调整学习率
        lr = adjust_learning_rate(
            optimizer=optimizer,
            gamma=args.gamma,
            epoch=epoch,
            step_index=step_index,
            iteration=iteration,
            epoch_size=epoch_size)

        # TODO load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        load_t1 = time.time()
        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f||' % (
                loss_l.item(),loss_c.item()) + 
                'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))
    #TODO 这里只保存了最后的训练模型结果，并没有保存中间训练过程中验证的最好结果
    torch.save(net.state_dict(), args.save_folder +
               'Final_' + args.version +'_' + args.dataset+ '.pth')


def adjust_learning_rate(optimizer, gamma, epoch,
                         step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5) 
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
