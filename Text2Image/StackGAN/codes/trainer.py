from __future__ import print_function

import sys

from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os

import time

import numpy as np
import torchfile

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init
from miscc.utils import save_img_results, save_model
from miscc.utils import KL_loss
from miscc.utils import compute_discriminator_loss, compute_generator_loss

from tensorboard import summary
# from tensorboard import FileWriter

"""
https://github.com/reedscot/icml2016
"""

class GANTrainer(object):
    def __init__(self, output_dir,device = 'cuda'):
        #TODO 如果是训练过程，则建立模型保存文件，图像和日志保存文件
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            # self.summary_writer = FileWriter(self.log_dir)
        #TODO 训练最大的迭代数
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        #TODO 使用的GPU情况
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.device = device
        if '1' in self.device:
            self.device_id = [1]
        else:
            self.device_id = [0]
        # torch.cuda.set_device(self.gpus[0])
        # cudnn.benchmark = True

    # ############# For training stageI GAN #############
    def load_network_stageI(self):
        from model import STAGE1_G, STAGE1_D
        netG = STAGE1_G(self.device)
        netG.apply(weights_init)
        # print(netG)
        netD = STAGE1_D()
        netD.apply(weights_init)
        # print(netD)

        if cfg.NET_G != '':
            state_dict = torch.load(cfg.NET_G,map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        if cfg.NET_D != '':
            state_dict = torch.load(cfg.NET_D,map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        if cfg.CUDA:
            netG.to(self.device)
            netD.to(self.device)
        return netG, netD

    # ############# For training stageII GAN  #############
    def load_network_stageII(self):
        from model import STAGE1_G, STAGE2_G, STAGE2_D

        Stage1_G = STAGE1_G(self.device)
        netG = STAGE2_G(Stage1_G)
        netG.apply(weights_init)
        # print(netG)
        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        elif cfg.STAGE1_G != '':
            state_dict = \
                torch.load(cfg.STAGE1_G,
                           map_location=lambda storage, loc: storage)
            netG.STAGE1_G.load_state_dict(state_dict)
            print('Load from: ', cfg.STAGE1_G)
        else:
            print("Please give the Stage1_G path")
            return

        netD = STAGE2_D()
        netD.apply(weights_init)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        # print(netD)

        if cfg.CUDA:
            netG.to(self.device)
            netD.to(self.device)
        return netG, netD

    def train(self, data_loader, stage=1):
        #TODO 根据不同的阶段加载不同的模型
        if stage == 1:
            netG, netD = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()

        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(
            torch.FloatTensor(batch_size, nz).normal_(mean=0, std=1)
        )
        #TODO 真实值以及假值
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        #TODO 加载到GPU设备上
        if cfg.CUDA:
            noise, fixed_noise = (noise.to(self.device),
                                  fixed_noise.to(self.device))
            real_labels, fake_labels = (real_labels.to(self.device),
                                        fake_labels.to(self.device))

        #TODO 定义生成器和判别器的优化函数
        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
        optimizerD = optim.Adam(netD.parameters(),
                       lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para,
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        count = 0
        for epoch in range(self.max_epoch):
            start_t = time.time()
            #TODO 更新学习率
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr

            for i, data in enumerate(data_loader, 0):
                ######################################################
                # TODO (1) Prepare training data
                ######################################################
                real_img_cpu, txt_embedding = data
                real_imgs = Variable(real_img_cpu)
                txt_embedding = Variable(txt_embedding)
                if cfg.CUDA:
                    real_imgs = real_imgs.to(self.device)
                    txt_embedding = txt_embedding.to(self.device)

                #######################################################
                # TODO  (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (txt_embedding, noise)
                _, fake_imgs, mu, logvar = nn.parallel.data_parallel(netG,
                                                                     inputs,
                                                                     device_ids=self.device_id)

                ############################
                # TODO  (3) Update D network
                ###########################
                netD.zero_grad()
                errD, errD_real, errD_wrong, errD_fake = \
                    compute_discriminator_loss(netD, real_imgs, fake_imgs,
                                               real_labels, fake_labels,
                                               conditions=mu, gpus=self.device_id)
                errD.backward()
                optimizerD.step()
                ############################
                # TODO (2) Update G network
                ###########################
                netG.zero_grad()
                errG = compute_generator_loss(netD, fake_imgs,
                                              real_labels, mu, self.device_id)
                kl_loss = KL_loss(mu, logvar)
                errG_total = errG + kl_loss * cfg.TRAIN.COEFF.KL
                errG_total.backward()
                optimizerG.step()

                count = count + 1

                sys.stdout.write(
                    "\r [Epoch %d/%d]  [D_loss %d  D_loss_real %d  D_loss_wrong %d  D_loss_fake %d]  [G_loss %d]  "
                    "[KL_loss %d]"%(
                        epoch,self.max_epoch,errD,errD_real,
                        errD_wrong,errD_fake,
                        errG,kl_loss.item()
                    )
                )

                if i % 100 == 0:
                    summary_D = summary.scalar('D_loss', errD.item())
                    summary_D_r = summary.scalar('D_loss_real', errD_real)
                    summary_D_w = summary.scalar('D_loss_wrong', errD_wrong)
                    summary_D_f = summary.scalar('D_loss_fake', errD_fake)
                    summary_G = summary.scalar('G_loss', errG.item())
                    summary_KL = summary.scalar('KL_loss', kl_loss.item())

                    # self.summary_writer.add_summary(summary_D, count)
                    # self.summary_writer.add_summary(summary_D_r, count)
                    # self.summary_writer.add_summary(summary_D_w, count)
                    # self.summary_writer.add_summary(summary_D_f, count)
                    # self.summary_writer.add_summary(summary_G, count)
                    # self.summary_writer.add_summary(summary_KL, count)

                    # save the image result for each epoch
                    inputs = (txt_embedding, fixed_noise)
                    lr_fake, fake, _, _ = nn.parallel.data_parallel(netG, inputs, self.device_id)
                    save_img_results(real_img_cpu, fake, epoch, self.image_dir)
                    if lr_fake is not None:
                        save_img_results(None, lr_fake, epoch, self.image_dir)
            end_t = time.time()
            print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                     Total Time: %.2fsec
                  '''
                  % (epoch, self.max_epoch, i, len(data_loader),
                     errD.item(), errG.item(), kl_loss.item(),
                     errD_real, errD_wrong,
                     errD_fake, (end_t - start_t)))
            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, epoch, self.model_dir)
        #
        save_model(netG, netD, self.max_epoch, self.model_dir)
        #
        # self.summary_writer.close()

    def sample(self, datapath, stage=1):
        if stage == 1:
            netG, _ = self.load_network_stageI()
        else:
            netG, _ = self.load_network_stageII()
        netG.eval()

        # TODO Load text embeddings generated from the encoder
        t_file = torchfile.load(datapath)
        captions_list = t_file.raw_txt
        embeddings = np.concatenate(t_file.fea_txt, axis=0)
        num_embeddings = len(captions_list)
        print('Successfully load sentences from: ', datapath)
        print('Total number of sentences:', num_embeddings)
        print('num_embeddings:', num_embeddings, embeddings.shape)
        # path to save generated samples
        save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')]
        mkdir_p(save_dir)

        batch_size = np.minimum(num_embeddings, self.batch_size)
        nz = cfg.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        if cfg.CUDA:
            noise = noise.to(self.device)
        count = 0
        while count < num_embeddings:
            if count > 3000:
                break
            iend = count + batch_size
            if iend > num_embeddings:
                iend = num_embeddings
                count = num_embeddings - batch_size
            embeddings_batch = embeddings[count:iend]
            # captions_batch = captions_list[count:iend]
            txt_embedding = Variable(torch.FloatTensor(embeddings_batch))
            if cfg.CUDA:
                txt_embedding = txt_embedding.to(self.device)

            #######################################################
            # TODO (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            inputs = (txt_embedding, noise)
            _, fake_imgs, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.device_id)
            #TODO 根据指定的batch size大小来生成图像的数量并保存起来
            for i in range(batch_size):
                save_name = '%s/%d.png' % (save_dir, count + i)
                im = fake_imgs[i].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                # print('im', im.shape)
                im = np.transpose(im, (1, 2, 0))
                # print('im', im.shape)
                im = Image.fromarray(im)
                im.save(save_name)
            count += batch_size

