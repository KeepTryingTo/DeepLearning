# -*- coding: utf-8 -*-


import os
import sys, os

sys.path.insert(0, os.path.join('..','..'))
# proj_root  = os.path.join('..','..')
proj_root = r'D:\conda3\Transfer_Learning\GANs\text-to-image\HDGan-master'
data_root  = os.path.join(proj_root, 'Data')
model_root = os.path.join(proj_root, 'Models')
save_root  =  os.path.join(proj_root, 'Results')
import numpy as np
import argparse, os
import torch, h5py
import torch.nn as nn
from collections import OrderedDict

from HDGan.proj_utils.local_utils import mkdirs
from HDGan.HDGan_test import test_gans
from HDGan.fuel.datasets import Dataset

from HDGan.models.hd_networks import Generator

if  __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Gans')    

    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='batch size.')
    parser.add_argument('--device_id', type=int, default= 0, 
                        help='which device')
    parser.add_argument('--load_from_epoch', type=int, default= 75,
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, default='HDGAN_256')
    parser.add_argument('--dataset',    type=str,  default= 'birds',
                        help='which dataset to use [birds or flowers]') 
    parser.add_argument('--noise_dim', type=int, default=100, metavar='N',
                        help='the dimension of noise.')
    parser.add_argument('--finest_size', type=int, default=256, metavar='N',
                        help='target image size.')
    parser.add_argument('--test_sample_num', type=int, default=4,
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--save_visual_results', action='store_true',
                        help='if save visual results in folders')

    args = parser.parse_args()
    args.cuda = False if torch.cuda.is_available() else False
    
    if args.finest_size <= 256:
        netG = Generator(sent_dim=1024, noise_dim=args.noise_dim,
                         emb_dim=128, hid_dim=128, num_resblock=1)
    else:
        from HDGan.models.hd_networks import GeneratorSuperL1Loss
        netG = GeneratorSuperL1Loss(sent_dim=1024, noise_dim=args.noise_dim,
                                    emb_dim=128, hid_dim=128, num_resblock=2)

    datadir = os.path.join(data_root, args.dataset)

    device_id = getattr(args, 'device_id', 0)
    
    if args.cuda:
        netG = netG.cuda(device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    dataset = Dataset(datadir, img_size=args.finest_size,
                      batch_size=args.batch_size, n_embed=1,
                      mode='test', multithread=False)
    model_name = args.model_name  
    
    save_folder  = os.path.join(save_root, args.dataset, model_name + '_testing_num_{}'.format(args.test_sample_num) )
    mkdirs(save_folder)
    
    test_gans(dataset, model_root, model_name, save_folder, netG, args)

    
    