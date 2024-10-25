import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime

from datasets.crowd import Crowd_qnrf, Crowd_nwpu, Crowd_sh
from models import vgg19
from losses.ot_loss import OT_Loss
from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils

from torch.nn import functional as F

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    gt_discretes = torch.stack(transposed_batch[2], 0)
    return images, points, gt_discretes


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        self.crop_size = args.crop_size
        sub_dir = 'input-{}_wot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}'.format(
            args.crop_size, args.wot, args.wtv, args.reg, args.num_of_iter_in_ot, args.norm_cood)
        #TODO 创建保存模型文件
        self.save_dir = os.path.join('ckpts', sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        #TODO 保存日志的文件
        time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        self.logger = log_utils.get_logger(os.path.join(self.save_dir, 'train-{:s}.log'.format(time_str)))
        log_utils.print_config(vars(args), self.logger)

        #TODO 是否使用GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            self.logger.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        #TODO 加载数据集
        downsample_ratio = 8
        if args.dataset.lower() == 'qnrf':
            self.datasets = {x: Crowd_qnrf(os.path.join(args.data_dir, x),
                                           args.crop_size, downsample_ratio, x) for x in ['train', 'val']}
        elif args.dataset.lower() == 'nwpu':
            self.datasets = {x: Crowd_nwpu(os.path.join(args.data_dir, x),
                                           args.crop_size, downsample_ratio, x) for x in ['train', 'val']}
        elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
            self.datasets = {'train': Crowd_sh(os.path.join(args.data_dir, 'train_data'),
                                               args.crop_size, downsample_ratio, 'train'),
                             'val': Crowd_sh(os.path.join(args.data_dir, 'test_data'),
                                             args.crop_size, downsample_ratio, 'val'),
                             }
        else:
            raise NotImplementedError

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers * self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        #TODO 加载模型
        self.model = vgg19()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            self.logger.info('loading pretrained model from ' + args.resume)
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))
        else:
            self.logger.info('random initialization')
        #TODO 定义最优化传输（Optimal Transport）损失
        #TODO 参数含义：裁剪图像大小，下采样比率，计算距离是否归一化坐标，加载设备，sinkhorn算法迭代次数，在sinkhorn算法中使用熵正则化
        self.ot_loss = OT_Loss(args.crop_size, downsample_ratio,
                               args.norm_cood, self.device,
                               args.num_of_iter_in_ot,
                               args.reg)
        #TODO 定义total variation loss，pixel-wise loss，Count Loss
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        #TODO 手动来处理当前列表中保存的结果，比如当前列表中保存的数量达到上限，就从0号索引删除一个，然后再加入列表中
        self.save_list = Save_Handle(max_num=1)
        #TODO 初始化MAE 和 MSE
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            self.epoch = epoch
            #TODO 首先开始训练
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                #TODO 验证开始
                self.val_epoch()

    def train_eopch(self):
        #TODO 记录训练过程中的损失记录以及评价结果
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        #TODO wasserstain distance
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        for step, (inputs, points, gt_discrete) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            #TODO 统计当前batch中每一张图像对应点标注的人群数
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            #TODO 根据加载的点标注加载到指定的设备
            points = [p.to(self.device) for p in points]
            #TODO 获取将坐标点转换为一维结构之后的表示并加载到设备上
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)

            with torch.set_grad_enabled(True):
                #TODO 输出密度图和归一化之后的密度图
                outputs, outputs_normed = self.model(inputs)
                #TODO Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points)
                ot_loss = ot_loss * self.args.wot
                ot_obj_value = ot_obj_value * self.args.wot
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)

                # TODO Compute counting loss.
                count_loss = self.mae(outputs.sum(1).sum(1).sum(1),
                                      torch.from_numpy(gd_count).float().to(self.device))
                epoch_count_loss.update(count_loss.item(), N)

                # TODO Compute TV loss.
                gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(
                    2).unsqueeze(3)
                #TODO 对真实的标注点进行归一化操作
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                tv_loss = (
                              self.tv_loss(
                                  outputs_normed, gt_discrete_normed
                              ).sum(1).sum(1).sum(1) * torch.from_numpy(gd_count).float().to(self.device)
                          ).mean(0) * self.args.wtv
                epoch_tv_loss.update(tv_loss.item(), N)

                loss = ot_loss + count_loss + tv_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)

        self.logger.info(
            'Epoch {} Train, Loss: {:.2f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, '
            'Count Loss: {:.2f}, TV Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                .format(self.epoch, epoch_loss.get_avg(), epoch_ot_loss.get_avg(), epoch_wd.get_avg(),
                        epoch_ot_obj_value.get_avg(), epoch_count_loss.get_avg(), epoch_tv_loss.get_avg(),
                        np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                        time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)

    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs, _ = self.model(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.logger.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time() - epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
            self.best_count += 1
    #TODO 由于直接将数据集加载模型中进行训练，很有可能显存不够，因此，采用滑动窗口的方式进行验证
    def myEval(self):
        epoch_res = []
        print('valing: ')
        epoch_start = time.time()
        for inputs, count,name in self.dataloaders['val']:
            with torch.no_grad():
                # nputs = cal_new_tensor(inputs, min_size=args.crop_size)
                inputs = inputs.to(self.device)
                crop_imgs, crop_masks = [], []
                b, c, h, w = inputs.size()
                # TODO 首先对图像进行裁剪，将裁剪之后的图像patch输入到模型中检测
                rh, rw = self.crop_size, self.crop_size
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                        mask = torch.zeros([b, 1, h, w]).to(self.device)
                        mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                        crop_masks.append(mask)
                # crop_imgs, crop_masks = map(
                #     lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks)
                # )
                crop_imgs = torch.cat(crop_imgs, dim=0)

                crop_preds = []
                nz, bz = crop_imgs.size(0), 1
                for i in range(0, nz, bz):
                    gs, gt = i, min(nz, i + bz)
                    crop_pred, _ = self.model(crop_imgs[gs:gt])

                    _, _, h1, w1 = crop_pred.size()
                    crop_pred = (
                            F.interpolate(
                                crop_pred,
                                size=(h1 * 8, w1 * 8),
                                mode="bilinear",
                                align_corners=True,
                            ) / 64
                    )

                    crop_preds.append(crop_pred)
                crop_preds = torch.cat(crop_preds, dim=0)

                # TODO splice them to the original size
                idx = 0
                pred_map = torch.zeros([b, 1, h, w]).to(self.device)
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                        idx += 1
                # for the overlapping area, compute average value
                # mask = crop_masks.sum(dim=0).unsqueeze(0)
                outputs = pred_map / mask

                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        self.logger.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time() - epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
            self.best_count += 1