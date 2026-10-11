#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp import Exp, check_exp_value, get_exp
from yolox.utils import (configure_module, configure_nccl,
                         configure_omp, get_num_devices)


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name",
                        type=str,
                        default=r'/home/ff/myProject/KGT/myProjects/myProjects/YOLOX-main/runs')
    parser.add_argument("-n", "--name", type=str,
                        default='yolox-s', help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str,help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",default=None,type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size",type=int, default=8, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=0, type=int, help="device for training"
    )
    parser.add_argument("-f","--exp_file",default=None,type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume",
        type=bool,
        default=True,
        help="resume training"
    )
    parser.add_argument("-c", "--ckpt",
                        default=r'/home/ff/myProject/KGT/myProjects/myProjects/YOLOX-main/weights/yolox_s.pth',
                        type=str, help="checkpoint file")
    parser.add_argument(
        "-e","--start_epoch",
        default=0,type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int,help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int,
        help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",dest="fp16",default=False,action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",type=str,nargs="?",const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o","--occupy",dest="occupy",default=False,action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument("-l","--logger",type=str,
        help="Logger to be used for metrics. \
                Implemented loggers include `tensorboard`, `mlflow` and `wandb`.",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    # configure_nccl() #TODO 配置多机器环境
    # configure_omp()
    cudnn.benchmark = True
    trainer = exp.get_trainer(args)
    trainer.train()


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    check_exp_value(exp)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    #TODO 获得可用GPU数量
    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )

"""
Average forward time: 2.51 ms, Average NMS time: 0.68 ms, Average inference time: 3.19 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.424
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.632
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.465
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.228
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.471
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.540
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.332
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.542
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.368
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.642
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.725
per class AP:
| class         | AP     | class        | AP     | class          | AP     |
|:--------------|:-------|:-------------|:-------|:---------------|:-------|
| person        | 52.943 | bicycle      | 33.058 | car            | 41.265 |
| motorcycle    | 43.252 | airplane     | 55.848 | bus            | 64.036 |
| train         | 63.217 | truck        | 38.461 | boat           | 27.545 |
| traffic light | 30.793 | fire hydrant | 62.993 | stop sign      | 68.248 |
| parking meter | 38.029 | bench        | 24.555 | bird           | 35.785 |
| cat           | 60.069 | dog          | 56.696 | horse          | 57.231 |
| sheep         | 52.989 | cow          | 54.291 | elephant       | 68.165 |
| bear          | 70.844 | zebra        | 64.671 | giraffe        | 68.128 |
| backpack      | 17.085 | umbrella     | 39.625 | handbag        | 13.032 |
| tie           | 35.437 | suitcase     | 39.978 | frisbee        | 63.340 |
| skis          | 25.200 | snowboard    | 36.407 | sports ball    | 43.087 |
| kite          | 43.916 | baseball bat | 32.828 | baseball glove | 39.202 |
| skateboard    | 47.725 | surfboard    | 40.936 | tennis racket  | 49.904 |
| bottle        | 36.788 | wine glass   | 40.595 | cup            | 43.135 |
| fork          | 30.248 | knife        | 18.309 | spoon          | 17.801 |
| bowl          | 43.600 | banana       | 30.346 | apple          | 26.660 |
| sandwich      | 37.984 | orange       | 34.934 | broccoli       | 28.911 |
| carrot        | 28.519 | hot dog      | 43.053 | pizza          | 54.995 |
| donut         | 59.493 | cake         | 43.470 | chair          | 32.683 |
| couch         | 45.837 | potted plant | 29.044 | bed            | 49.842 |
| dining table  | 34.521 | toilet       | 67.496 | tv             | 57.656 |
| laptop        | 61.360 | mouse        | 53.669 | remote         | 30.742 |
| keyboard      | 51.804 | cell phone   | 34.256 | microwave      | 58.608 |
| oven          | 38.997 | toaster      | 15.076 | sink           | 39.321 |
| refrigerator  | 53.424 | book         | 14.834 | clock          | 51.710 |
| vase          | 42.190 | scissors     | 32.758 | teddy bear     | 48.279 |
| hair drier    | 10.315 | toothbrush   | 17.765 |                |        |
per class AR:
| class         | AR     | class        | AR     | class          | AR     |
|:--------------|:-------|:-------------|:-------|:---------------|:-------|
| person        | 62.132 | bicycle      | 48.560 | car            | 54.830 |
| motorcycle    | 55.149 | airplane     | 66.212 | bus            | 72.150 |
| train         | 72.185 | truck        | 60.321 | boat           | 44.510 |
| traffic light | 45.305 | fire hydrant | 70.446 | stop sign      | 76.691 |
| parking meter | 49.176 | bench        | 44.453 | bird           | 47.612 |
| cat           | 72.732 | dog          | 68.016 | horse          | 66.456 |
| sheep         | 62.847 | cow          | 63.612 | elephant       | 76.245 |
| bear          | 78.680 | zebra        | 73.964 | giraffe        | 75.139 |
| backpack      | 42.325 | umbrella     | 55.420 | handbag        | 38.208 |
| tie           | 48.908 | suitcase     | 60.924 | frisbee        | 71.412 |
| skis          | 43.046 | snowboard    | 50.126 | sports ball    | 50.737 |
| kite          | 57.350 | baseball bat | 49.922 | baseball glove | 51.376 |
| skateboard    | 58.147 | surfboard    | 53.202 | tennis racket  | 60.633 |
| bottle        | 53.203 | wine glass   | 52.164 | cup            | 59.835 |
| fork          | 47.200 | knife        | 38.951 | spoon          | 39.709 |
| bowl          | 65.054 | banana       | 52.085 | apple          | 53.095 |
| sandwich      | 62.457 | orange       | 59.684 | broccoli       | 56.394 |
| carrot        | 53.847 | hot dog      | 59.891 | pizza          | 67.191 |
| donut         | 71.160 | cake         | 61.903 | chair          | 53.336 |
| couch         | 67.130 | potted plant | 52.115 | bed            | 68.055 |
| dining table  | 59.859 | toilet       | 79.850 | tv             | 72.635 |
| laptop        | 71.319 | mouse        | 63.800 | remote         | 50.655 |
| keyboard      | 67.704 | cell phone   | 51.326 | microwave      | 71.280 |
| oven          | 63.064 | toaster      | 54.744 | sink           | 58.117 |
| refrigerator  | 69.921 | book         | 38.828 | clock          | 65.081 |
| vase          | 60.155 | scissors     | 50.703 | teddy bear     | 63.154 |
| hair drier    | 16.757 | toothbrush   | 39.540 |                |        |

"""
