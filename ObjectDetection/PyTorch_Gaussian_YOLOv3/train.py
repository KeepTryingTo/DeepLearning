from __future__ import division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils.utils import *
from utils.seed import set_seed, setup_cudnn
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.parse_yolo_weights import parse_yolo_weights
from models.yolov3 import *
from dataset.cocodataset import *


import argparse
import yaml
import random

import torch
from torch.autograd import Variable
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/gaussian_yolov3_default.cfg',
                        help='config file. see readme')
    parser.add_argument('--weights_path', type=str,
                        default=r'./requirements/darknet53.conv.74',
                        help='darknet weights file')
    parser.add_argument('--n_cpu', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--checkpoint_interval', type=int,
                        default=1000, help='interval between saving checkpoints')
    parser.add_argument('--eval_interval', type=int,
                            default=4000, help='interval between evaluations')
    parser.add_argument('--checkpoint',
                        type=str,
                        default=r'./checkpoints/snapshot18000.ckpt',
                        help='pytorch checkpoint file path')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./checkpoints',
                        help='directory where checkpoint files are saved')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument(
        '--tfboard_dir', help='tensorboard path for logging',
        type=str, default=None)
    return parser.parse_args()


def main():
    """
    YOLOv3 trainer. See README for details.
    """
    args = parse_args()
    print("Setting Arguments.. : ", args)

    cuda = torch.cuda.is_available() and args.use_cuda
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Parse config settings
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f,yaml.Loader)

    print("successfully loaded config file: ", cfg)

    momentum = cfg['TRAIN']['MOMENTUM']
    decay = cfg['TRAIN']['DECAY']
    burn_in = cfg['TRAIN']['BURN_IN']
    iter_size = cfg['TRAIN']['MAXITER']
    steps = eval(cfg['TRAIN']['STEPS'])
    batch_size = cfg['TRAIN']['BATCHSIZE']
    subdivision = cfg['TRAIN']['SUBDIVISION']
    ignore_thre = cfg['TRAIN']['IGNORETHRE']
    random_resize = cfg['AUGMENTATION']['RANDRESIZE']
    base_lr = cfg['TRAIN']['LR'] / batch_size / subdivision
    gradient_clip = cfg['TRAIN']['GRADIENT_CLIP']

    print('effective_batch_size = batch_size * iter_size = %d * %d' %
          (batch_size, subdivision))

    # Make trainer behavior deterministic
    set_seed(seed=0)
    setup_cudnn(deterministic=True)

    # Learning rate setup
    def burnin_schedule(i):
        if i < burn_in:
            factor = pow(i / burn_in, 4)
        elif i < steps[0]:
            factor = 1.0
        elif i < steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    #TODO  Initiate model
    model = YOLOv3(cfg['MODEL'], ignore_thre=ignore_thre)
    global state
    if args.weights_path:
        print("loading darknet weights....", args.weights_path)
        parse_yolo_weights(model, args.weights_path)
    if args.checkpoint:
        print("loading pytorch ckpt...", args.checkpoint)
        state = torch.load(args.checkpoint)
        if 'model_state_dict' in state.keys():
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
    print('load model is done ...')
    if cuda:
        print("using cuda") 
        model = model.cuda()

    if args.tfboard_dir:
        print("using tfboard")
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(args.tfboard_dir)

    model.train()

    root_dir = r'/home/ff/myProject/KGT/myProjects/myDataset/coco'
    imgsize = cfg['TRAIN']['IMGSIZE']
    dataset = COCODataset(model_type=cfg['MODEL']['TYPE'],
                  data_dir=root_dir,
                  img_size=imgsize,
                  augmentation=cfg['AUGMENTATION'],
                  debug=args.debug)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)
    dataiterator = iter(dataloader)

    evaluator = COCOAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
                    data_dir=root_dir,
                    img_size=cfg['TEST']['IMGSIZE'],
                    confthre=cfg['TEST']['CONFTHRE'],
                    nmsthre=cfg['TEST']['NMSTHRE'])

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # optimizer setup
    # set weight decay only on conv.weight
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'conv.weight' in key:
            params += [{'params':value, 'weight_decay':decay * batch_size * subdivision}]
        else:
            params += [{'params':value, 'weight_decay':0.0}]
    optimizer = optim.SGD(params,
                          lr=base_lr,
                          momentum=momentum,
                          dampening=0,
                          weight_decay=decay * batch_size * subdivision)

    iter_state = 0

    if args.checkpoint:
        if 'optimizer_state_dict' in state.keys():
            optimizer.load_state_dict(state['optimizer_state_dict'])
            iter_state = state['iter'] + 1

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    # start training loop
    for iter_i in range(iter_state, iter_size + 1):

        # COCO evaluation
        if iter_i % args.eval_interval == 0 and iter_i > 0:
            print('evaluating...')
            ap = evaluator.evaluate(model)
            model.train()
            print('AP50:{:.3f} \t AP75:{:.3f} \t AP5095: {:.3f} \t'
                  'AP5095_S: {:.3f} \t AP5095_M: {:.3f} \t AP5095: {:.3f}'.format(
                ap['aP50'],ap['aP75'],ap['aP5095'],ap['aP5095_S'],
                ap['aP5095_M'],ap['aP5095_L']
            ))
            if args.tfboard_dir:
                # val/aP
                tblogger.add_scalar('val/aP50', ap['aP50'], iter_i)
                tblogger.add_scalar('val/aP75', ap['aP75'], iter_i)
                tblogger.add_scalar('val/aP5095', ap['aP5095'], iter_i)
                tblogger.add_scalar('val/aP5095_S', ap['aP5095_S'], iter_i)
                tblogger.add_scalar('val/aP5095_M', ap['aP5095_M'], iter_i)
                tblogger.add_scalar('val/aP5095_L', ap['aP5095_L'], iter_i)

        # TODO subdivision loop
        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            try:
                imgs, targets, _, _ = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, targets, _, _ = next(dataiterator)  # load a batch
            imgs = Variable(imgs.type(dtype))
            targets = Variable(targets.type(dtype), requires_grad=False)
            loss = model(imgs, targets)
            loss.backward()

        if gradient_clip >= 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), gradient_clip)

        optimizer.step()
        scheduler.step()

        if iter_i % 1 == 0:
            # logging
            current_lr = scheduler.get_lr()[0] * batch_size * subdivision
            print('[Iter %d/%d] [lr %f] '
                  '[Losses: xy %f, wh %f, conf %f, cls %f, total %f, imgsize %d]'
                  % (iter_i, iter_size, current_lr,
                     model.loss_dict['xy'], model.loss_dict['wh'],
                     model.loss_dict['conf'], model.loss_dict['cls'], 
                     loss, imgsize),
                  flush=True)

            if args.tfboard_dir:
                # lr
                tblogger.add_scalar('lr', current_lr, iter_i)
                # train/loss
                tblogger.add_scalar('train/loss_xy', model.loss_dict['xy'], iter_i)
                tblogger.add_scalar('train/loss_wh', model.loss_dict['wh'], iter_i)
                tblogger.add_scalar('train/loss_conf', model.loss_dict['conf'], iter_i)
                tblogger.add_scalar('train/loss_cls', model.loss_dict['cls'], iter_i)
                tblogger.add_scalar('train/loss', loss, iter_i)

            # random resizing
            if random_resize:
                imgsize = (random.randint(0, 9) % 10 + 10) * 32
                dataset.img_shape = (imgsize, imgsize)
                dataset.img_size = imgsize
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)
                dataiterator = iter(dataloader)

        # save checkpoint
        if args.checkpoint_dir and iter_i > 0 and (iter_i % args.checkpoint_interval == 0):
            torch.save({'iter': iter_i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        },
                        os.path.join(args.checkpoint_dir, "snapshot"+str(iter_i)+".ckpt"))

    if args.tfboard_dir:
        tblogger.close()


if __name__ == '__main__':
    main()

"""
DONE (t=46.00s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.168
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.333
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.153
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.044
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.156
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.250
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.170
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.093
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.371
AP50:0.333 	 AP75:0.153 	 AP5095: 0.168 	AP5095_S: 0.044 	 AP5095_M: 0.156 	 AP5095: 0.250
"""