# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import sys
import torch
sys.path.insert(0, os.path.join('..', '..'))

proj_root = os.path.join('..', '..')
data_root = os.path.join(proj_root, 'Data')
model_root = os.path.join(proj_root, 'Models')

import torch.nn as nn
from collections import OrderedDict

from HDGan.models.hd_networks import Generator
from HDGan.models.hd_networks import Discriminator

from HDGan.HDGan import train_gans
from HDGan.fuel.datasets import Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gans')

    parser.add_argument('--reuse_weights',   action='store_true',
                        default=False, help='continue from last checkout point')
    parser.add_argument('--load_from_epoch', type=int,default=0,  help='load from epoch')

    parser.add_argument('--batch_size', type=int,default=2, metavar='N', help='batch size.')
    parser.add_argument('--device_id',  type=int,default=0,  help='which device')

    parser.add_argument('--model_name', type=str,      default=None)
    parser.add_argument('--dataset',    type=str,      default=r'birds',
                        help='which dataset to use [birds or flowers]')

    parser.add_argument('--num_resblock', type=int, default=1,help='number of resblock in generator')
    parser.add_argument('--epoch_decay', type=float, default=100,
                        help='decay learning rate by half every epoch_decay')
    parser.add_argument('--finest_size', type=int, default=256,metavar='N', help='target image size.')
    parser.add_argument('--init_256generator_from', type=str,  default='')
    parser.add_argument('--maxepoch', type=int, default=600, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--g_lr', type=float, default=0.0002, metavar='LR',help='learning rate (default: 0.01)')
    parser.add_argument('--d_lr', type=float, default=0.0002, metavar='LR',help='learning rate (default: 0.01)')
    parser.add_argument('--save_freq', type=int, default=5, metavar='N',help='how frequent to save the model')
    parser.add_argument('--display_freq', type=int, default=200, metavar='N',help='plot the results every {} batches')
    parser.add_argument('--verbose_per_iter', type=int, default=10,help='print losses per iteration')
    parser.add_argument('--num_emb', type=int, default=4, metavar='N',
                        help='number of emb chosen for each image during training.')
    parser.add_argument('--noise_dim', type=int, default=100, metavar='N',help='the dimension of noise.')
    parser.add_argument('--ncritic', type=int, default=1, metavar='N',help='the channel of each image.')
    parser.add_argument('--test_sample_num', type=int, default=4,
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--KL_COE', type=float, default=4, metavar='N',help='kl divergency coefficient.')
    parser.add_argument('--visdom_port', type=int, default=43426,
                        help='The port should be the same with the port when launching visdom')
    parser.add_argument('--gpus', type=str, default='0', help='which gpu')
    # add more
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print(args)
    
    # TODO 加载生成器模型Generator
    if args.finest_size <= 256:
        netG = Generator(sent_dim=1024, noise_dim=args.noise_dim,
                         emb_dim=128, hid_dim=128, num_resblock=1)
    else:
        assert args.init_256generator_from != '', '256 generator need to be intialized'
        from HDGan.models.hd_networks import GeneratorSuperL1Loss
        netG = GeneratorSuperL1Loss(sent_dim=1024, noise_dim=args.noise_dim,
                                    emb_dim=128, hid_dim=128, num_resblock=2,
                                    G256_weightspath=args.init_256generator_from)
    # TODO Discriminator
    netD = Discriminator(num_chan=3, hid_dim=128, sent_dim=1024, emb_dim=128)

    gpus = [a for a in range(len(args.gpus.split(',')))]
    torch.cuda.set_device(gpus[0])
    args.batch_size = args.batch_size * len(gpus)
    if args.cuda:
        print ('>> Parallel models in {} GPUS'.format(gpus))
        netD = nn.parallel.DataParallel(netD, device_ids=range(len(gpus)))
        netG = nn.parallel.DataParallel(netG, device_ids=range(len(gpus)))

        netD = netD.cuda()
        netG = netG.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    data_root = r'D:\conda3\Transfer_Learning\GANs\text-to-image\HDGan-master\Data'
    data_name = args.dataset
    datadir = os.path.join(data_root, data_name)
    #TODO num_emb表示对每一张图像所选择的词嵌入句子向量数，因为每一张图像都有对应十个句子进行描述
    dataset_train = Dataset(datadir, img_size=args.finest_size,
                            batch_size=args.batch_size, n_embed=args.num_emb, mode='train')
    dataset_test = Dataset(datadir, img_size=args.finest_size,
                           batch_size=args.batch_size, n_embed=1, mode='test')

    model_name = '{}_{}'.format(args.model_name, data_name)

    print('>> Start training ...')
    train_gans((dataset_train, dataset_test), model_root, model_name, netG, netD, args)
