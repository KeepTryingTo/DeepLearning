import os
import yaml
import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from torch import nn

from torch.utils.data.distributed import DistributedSampler
from datasets.coco import COCODataSets
from nets.centernet import CenterNet
from losses.centernet_loss import CenterNetLoss
from torch.utils.data.dataloader import DataLoader
from commons.model_utils import (rand_seed, is_parallel, ModelEMA,
                                 freeze_bn, reduce_sum)
from metrics.map import coco_map
from torch.nn.functional import interpolate
from commons.optims_utils import (WarmUpCosineDecayMultiStepLRAdjust,
                                  split_optimizer)
from commons.centernet_gt import CenterNetGT

rand_seed(1024)


class DDPApexProcessor(object):
    def __init__(self, cfg_path):
        with open(cfg_path, 'r') as rf:
            self.cfg = yaml.safe_load(rf)
        #TODO 数据配置信息
        self.data_cfg = self.cfg['data']
        #TODO 模型配置信息
        self.model_cfg = self.cfg['model']
        #TODO 优化器参数
        self.optim_cfg = self.cfg['optim']
        #TODO 超参数配置信息
        self.hyper_params = self.cfg['hyper_params']
        self.val_cfg = self.cfg['val']
        print(self.data_cfg)
        print(self.model_cfg)
        print(self.optim_cfg)
        print(self.hyper_params)
        print(self.val_cfg)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cfg['gpus'])
        self.gpu_num = len(str(self.cfg['gpus']).split(","))
        # dist.init_process_group(backend='nccl')
        ###########################################################################################
        self.tdata = COCODataSets(img_root=self.data_cfg['train_img_root'],
                                  annotation_path=self.data_cfg['train_annotation_path'],
                                  img_size=self.data_cfg['img_size'],
                                  debug=self.data_cfg['debug'],
                                  use_crowd=self.data_cfg['use_crowd'],
                                  augments=True,
                                  remove_blank=self.data_cfg['remove_blank']
                                  )
        self.tloader = DataLoader(dataset=self.tdata,
                                  batch_size=self.data_cfg['batch_size'],
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.tdata.collate_fn,
                                  # sampler=DistributedSampler(dataset=self.tdata, shuffle=True)
                                  )
        self.vdata = COCODataSets(img_root=self.data_cfg['val_img_root'],
                                  annotation_path=self.data_cfg['val_annotation_path'],
                                  img_size=self.data_cfg['img_size'],
                                  debug=self.data_cfg['debug'],
                                  use_crowd=self.data_cfg['use_crowd'],
                                  augments=False,
                                  remove_blank=False
                                  )
        self.vloader = DataLoader(dataset=self.vdata,
                                  batch_size=1,
                                  num_workers=1,
                                  collate_fn=self.vdata.collate_fn,
                                  # sampler=DistributedSampler(dataset=self.vdata, shuffle=False)
                                  )
        print("train_data: ", len(self.tdata), " | ",
              "val_data: ", len(self.vdata), " | ",
              "empty_data: ", self.tdata.empty_images_len)
        print("train_iter: ", len(self.tloader), " | ",
              "val_iter: ", len(self.vloader))
        ############################################################################################
        model = CenterNet(num_cls=self.model_cfg['num_cls'],
                          PIXEL_MEAN=self.model_cfg['PIXEL_MEAN'],
                          PIXEL_STD=self.model_cfg['PIXEL_STD'],
                          backbone=self.model_cfg['backbone'],
                          cfg=self.model_cfg)
        print('load model is done ...')

        self.best_map = 0.
        self.best_map50 = 0.
        #TODO 定义优化器
        optimizer = split_optimizer(model, self.optim_cfg)
        # local_rank = dist.get_rank()
        self.local_rank = 0
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        model.to(self.device)
        self.model = model
        # if self.optim_cfg['sync_bn']:
        #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # self.model = nn.parallel.distributed.DistributedDataParallel(model,
        #                                                              device_ids=[local_rank],
        #                                                              output_device=local_rank)
        self.optimizer = optimizer
        self.ema = ModelEMA(self.model)

        self.gt_generator=CenterNetGT(alpha=self.model_cfg['alpha'],
                                      beta=self.model_cfg['beta'],
                                      num_cls=self.model_cfg['num_cls'],
                                      wh_planes=self.model_cfg['wh_planes'],
                                      down_ratio=self.model_cfg['down_ratio'],
                                      wh_area_process=self.model_cfg['wh_area_process'])
        print('generate ground truth ...')
        self.creterion = CenterNetLoss(hm_weight=self.hyper_params['hm_weight'],
                                       wh_weight=self.hyper_params['wh_weight'],
                                       down_ratio=self.model_cfg['down_ratio'])

        self.lr_adjuster = WarmUpCosineDecayMultiStepLRAdjust(init_lr=self.optim_cfg['lr'],
                                                              milestones=self.optim_cfg['milestones'],
                                                              warm_up_epoch=self.optim_cfg['warm_up_epoch'],
                                                              iter_per_epoch=len(self.tloader),
                                                              epochs=self.optim_cfg['epochs'],
                                                              cosine_weights=self.optim_cfg['cosine_weights']
                                                              )

    def train(self, epoch):
        self.model.train()
        if self.model_cfg['freeze_bn']:
            self.model.apply(freeze_bn)
        if self.local_rank == 0:
            pbar = tqdm(self.tloader)
        else:
            pbar = self.tloader

        loss_list = [list(), list(), list()]
        lr = 0

        for i, (img_tensor, targets_tensor, _) in enumerate(pbar):
            if len(self.hyper_params['multi_scale']) > 2:
                target_size = np.random.choice(self.hyper_params['multi_scale'])
                img_tensor = interpolate(img_tensor, mode='bilinear', size=target_size, align_corners=False)
            bs, _, h, w = img_tensor.shape
            with torch.no_grad():
                targets_tensor[:, 3:] = targets_tensor[:, 3:] * torch.tensor(data=[w, h, w, h])
                img_tensor = img_tensor.to(device=self.device)
                heatmaps, box_targets, reg_weights=self.gt_generator(targets_tensor, [h,w], bs)

            self.optimizer.zero_grad()
            cls_pred,wh_pred = self.model(img_tensor)
            total_loss, detail_loss = self.creterion(cls_pred,wh_pred,heatmaps, box_targets, reg_weights)
            total_loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.optim_cfg['max_norm'],
            #                          norm_type=2)
            self.lr_adjuster(self.optimizer, i, epoch)
            lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.step()
            self.ema.update(self.model)
            loss_cls, loss_wh = detail_loss
            loss_list[0].append(total_loss.item())
            loss_list[1].append(loss_cls.item())
            loss_list[2].append(loss_wh.item())
            if self.local_rank == 0:
                pbar.set_description(
                    "epoch:{:2d} | size:{:3d} | target_loss:{:6.4f} | loss_cls:{:6.4f} | loss_wh:{:6.4f} | lr:{:8.6f}".format(
                        epoch + 1,
                        h,
                        total_loss.item(),
                        loss_cls.item(),
                        loss_wh.item(),
                        lr
                    ))
        self.ema.update_attr(self.model)
        mean_loss_list = [np.array(item).mean() for item in loss_list]
        print(
            "epoch:{:3d} | local:{:3d} | target_loss:{:6.4f} | loss_cls:{:6.4f} | loss_wh:{:6.4f} | lr:{:8.6f}"
                .format(epoch + 1,
                        self.local_rank,
                        mean_loss_list[0],
                        mean_loss_list[1],
                        mean_loss_list[2],
                        lr))

    @torch.no_grad()
    def val(self, epoch):
        predict_list = list()
        target_list = list()
        self.model.eval()
        self.ema.ema.eval()
        if self.local_rank == 0:
            pbar = tqdm(self.vloader)
        else:
            pbar = self.vloader
        for img_tensor, targets_tensor, _ in pbar:
            _, _, h, w = img_tensor.shape
            targets_tensor[:, 3:] = targets_tensor[:, 3:] * torch.tensor(data=[w, h, w, h])
            img_tensor = img_tensor.to(self.device)
            targets_tensor = targets_tensor.to(self.device)
            predicts = self.ema.ema(img_tensor)  #list(predict) predict.shape=[num_box,6]  6==>x1,y1,x2,y2,score,label
            for i, predict in enumerate(predicts):
                predict[:, [0, 2]] = predict[:, [0, 2]].clamp(min=0, max=w)
                predict[:, [1, 3]] = predict[:, [1, 3]].clamp(min=0, max=h)
                predict_list.append(predict)
                targets_sample = targets_tensor[targets_tensor[:, 0] == i][:, 2:]
                target_list.append(targets_sample)
        mp, mr, map50, map = coco_map(predict_list, target_list)
        mp = reduce_sum(torch.tensor(mp, device=self.device)).item() / self.gpu_num
        mr = reduce_sum(torch.tensor(mr, device=self.device)).item() / self.gpu_num
        map50 = reduce_sum(torch.tensor(map50, device=self.device)).item() / self.gpu_num
        map = reduce_sum(torch.tensor(map, device=self.device)).item() / self.gpu_num
        print("epoch: {:2d}|local:{:d}|mp:{:6.4f}|mr:{:6.4f}|map50:{:6.4f}|map:{:6.4f}"
              .format(epoch + 1,
                      self.local_rank,
                      mp * 100,
                      mr * 100,
                      map50 * 100,
                      map * 100))
        last_weight_path = os.path.join(self.val_cfg['weight_path'],
                                        "{:d}_{:s}_last.pth"
                                        .format(self.local_rank, self.cfg['model_name']))
        best_map_weight_path = os.path.join(self.val_cfg['weight_path'],
                                            "{:d}_{:s}_best_map.pth"
                                            .format(self.local_rank, self.cfg['model_name']))
        best_map50_weight_path = os.path.join(self.val_cfg['weight_path'],
                                              "{:d}_{:s}_best_map50.pth"
                                              .format(self.local_rank, self.cfg['model_name']))


        ema_static = self.ema.ema.state_dict()
        cpkt = {
            "ema": ema_static,
            "map": map * 100,
            "epoch": epoch,
            "map50": map50 * 100
        }
        if self.local_rank != 0:
            return
        torch.save(cpkt, last_weight_path)
        if map > self.best_map:
            torch.save(cpkt, best_map_weight_path)
            self.best_map = map
        if map50 > self.best_map50:
            torch.save(cpkt, best_map50_weight_path)
            self.best_map50 = map50

    def run(self):
        for epoch in range(self.optim_cfg['epochs']):
            self.train(epoch)
            if (epoch + 1) % self.val_cfg['interval'] == 0:
                self.val(epoch)
