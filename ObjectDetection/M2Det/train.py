from __future__ import print_function
import os
import warnings
warnings.filterwarnings('ignore')

from m2det import build_net
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from layers.functions import PriorBox
from utils.core import *

parser = argparse.ArgumentParser(description='M2Det Training')
parser.add_argument('-c', '--config', default='configs/m2det320_vgg.py')
parser.add_argument('-d', '--dataset', default='VOC', help='VOC or COCO dataset')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-t', '--tensorboard', type=bool, default=False, help='Use tensorborad to show the Loss Graph')
args = parser.parse_args()

print_info('----------------------------------------------------------------------\n'
           '|                       M2Det Training Program                       |\n'
           '----------------------------------------------------------------------',['yellow','bold'])

logger = set_logger(args.tensorboard)

#TODO 加载配置文件
cfg = Config.fromfile(args.config)
net = build_net('train', 
                size = cfg.model.input_size, # Only 320, 512, 704 and 800 are supported
                config = cfg.model.m2det_config)
init_net(net, cfg, args.resume_net) # init the network with pretrained weights or resumed weights

#TODO 是否采用分布式计算
if args.ngpu>1:
    net = torch.nn.DataParallel(net)
if cfg.train_cfg.cuda:
    net.cuda()
    cudnn.benchmark = True

optimizer = set_optimizer(net, cfg)
criterion = set_criterion(cfg)

device = 'cuda' if cfg.train_cfg.cuda else 'cpu'
priorbox = PriorBox(anchors(cfg),device= device)

with torch.no_grad():
    priors = priorbox.forward()
    if cfg.train_cfg.cuda:
        priors = priors.cuda()

if __name__ == '__main__':
    #TODO 训练模式
    net.train()
    epoch = args.resume_epoch
    #TODO 读取数据集
    print_info('===> Loading Dataset...',['yellow','bold'])
    dataset = get_dataloader(cfg, args.dataset, 'train_sets')

    epoch_size = len(dataset) // (cfg.train_cfg.per_batch_size * args.ngpu)
    max_iter = getattr(cfg.train_cfg.step_lr,args.dataset)[-1] * epoch_size
    #TODO 根据指定的学习率调整阶段
    stepvalues = [_*epoch_size for _ in getattr(
            cfg.train_cfg.step_lr, args.dataset
        )[:-1]
    ]
    print_info('===> Training M2Det on ' + args.dataset, ['yellow','bold'])

    if cfg.train_cfg.cuda:
        print_info('using GPU')

    #TODO
    if cfg.train_cfg.cuda:
        generator = torch.Generator(device='cuda')
    else:
        generator = torch.Generator(device='cpu')

    step_index = 0
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    for iteration in range(start_iter, max_iter):

        if iteration % epoch_size == 0:
            #TODO 加载数据集
            batch_iterator = iter(
                data.DataLoader(dataset,
                              cfg.train_cfg.per_batch_size * args.ngpu,
                              shuffle=True,
                              generator=generator,
                              num_workers=cfg.train_cfg.num_workers,
                              collate_fn=detection_collate))
            #TODO 保存模型
            if epoch % cfg.model.save_eposhs == 0:
                save_checkpoint(net, cfg, final=False,
                                datasetname = args.dataset,
                                epoch=epoch)
            epoch += 1
        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        #TODO 调整学习率
        lr = adjust_learning_rate(optimizer=optimizer, gamma=cfg.train_cfg.gamma,
                                  epoch=epoch, step_index=step_index,
                                  iteration=iteration,
                                  epoch_size=epoch_size, cfg=cfg)
        images, targets = next(batch_iterator)
        #TODO 加载到GPU上
        if cfg.train_cfg.cuda:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
        out = net(images)

        #TODO 梯度反向传播
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        write_logger({'loc_loss':loss_l.item(),
                      'conf_loss':loss_c.item(),
                      'loss':loss.item()},logger,iteration,status=args.tensorboard)
        loss.backward()
        optimizer.step()

        load_t1 = time.time()
        print_train_log(iteration, cfg.train_cfg.print_epochs,
                            [time.ctime(),epoch,
                             iteration%epoch_size,
                             epoch_size,iteration,
                             loss_l.item(),loss_c.item(),
                             load_t1-load_t0,lr]
                        )
    save_checkpoint(net, cfg, final=True,
                    datasetname=args.dataset,epoch=-1)
