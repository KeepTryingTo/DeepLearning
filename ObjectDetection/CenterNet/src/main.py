from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from src.lib.opts import opts
from src.lib.models.model import create_model, load_model, save_model
from src.lib.models.data_parallel import DataParallel
from src.lib.logger import Logger
from src.lib.datasets.dataset_factory import get_dataset
from src.lib.trains.train_factory import train_factory


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  #TODO 构建数据集
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

  #TODO 构建模型
  print('Creating model and head_conv: {} ...'.format(opt.head_conv))
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  # print('model: {}'.format(model))
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model=model, model_path=opt.load_model,
      optimizer=optimizer, resume=opt.resume,
      lr=opt.lr, lr_step=opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  #TODO 默认最开始会进行测试
  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training and val intervals {}...'.format(opt.val_intervals))
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    #TODO 进行模型的训练
    log_dict_train, _ = trainer.train(epoch, train_loader)
    #TODO 保存日志记录结果
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    #TODO 保存模型 val_intervals=5
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      #TODO 进行验证
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      #TODO 记录日志结果
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      #TODO 保存最好的模型结果
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
        #TODO 保存最后一个epochs模型
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    #TODO 保存指定epochs位置模型
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)