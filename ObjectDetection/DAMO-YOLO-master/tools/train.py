#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import os
import argparse
from loguru import logger

import torch
import torch.distributed as dist

from damo.apis import Trainer
from damo.config.base import parse_config
from damo.utils import synchronize


def make_parser():
    """
    Create a parser with some common arguments used by users.

    Returns:
        argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser('Damo-Yolo train parser')

    parser.add_argument(
        '-f','--config_file',
        default=r'/home/ff/myProject/KGT/myProjects/myProjects/DAMO-YOLO-master/configs/damoyolo_tinynasL20_T.py',
        type=str,
        help='plz input your config file',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tea_config', type=str, default=None)
    parser.add_argument('--tea_ckpt', type=str, default=None)
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    torch.cuda.set_device(args.local_rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    dist.init_process_group(backend='gloo', init_method='env://', rank=0,
                            world_size=int(os.environ['WORLD_SIZE']
                                           ) if 'WORLD_SIZE' in os.environ else 1)

    synchronize()
    if args.tea_config is not None:
        tea_config = parse_config(args.tea_config)
    else:
        tea_config = None

    config = parse_config(args.config_file)
    config.merge(args.opts)


    trainer = Trainer(config, args, tea_config)
    trainer.train(args.local_rank)


if __name__ == '__main__':
    main()
