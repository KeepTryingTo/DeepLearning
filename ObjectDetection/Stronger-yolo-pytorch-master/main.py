
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from models import *
from trainers import *
import json
from yacscfg import _C as cfg
from torch import optim
import argparse
import torch
def main(args):
    gpus=[str(g) for g in args.devices]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
    net = eval(cfg.MODEL.modeltype)(cfg=args.MODEL).cuda()
    optimizer = optim.Adam(net.parameters(),lr=args.OPTIM.lr_initial)
    scheduler=optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.OPTIM.milestones, gamma=0.1)
    #TODO 默认针对VOC数据集进行训练
    _Trainer = eval('Trainer_{}'.format(args.DATASET.dataset))(args=args,
                       model=net,
                       optimizer=optimizer,
                       lrscheduler=scheduler
                       )
    if args.do_test:
      _Trainer._valid_epoch(validiter=-1)
    else:
      # _Trainer._valid_epoch(validiter=-1)
      _Trainer.train()

  #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DEMO configuration")
    parser.add_argument(
        "--config-file",
        # default = 'configs/strongerv3_kl.yaml'
        default=r'./configs/coco.yaml'
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:1'
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)
    main(args=cfg)
"""
VOC:
mAP:0.750534696916979
AP@aeroplane:0.8275318619674217
AP@bicycle:0.8326714962764745
AP@bird:0.7001646628769136
AP@boat:0.6402033449288956
AP@bottle:0.6371673007280283
AP@bus:0.8387599295788526
AP@car:0.8687864637712742
AP@cat:0.851489979203143
AP@chair:0.5744246078444702
AP@cow:0.7135517985336831
AP@diningtable:0.7053429181355869
AP@dog:0.7991225296080589
AP@horse:0.8491678194436082
AP@motorbike:0.8143549411856881
AP@person:0.8309471243852518
AP@pottedplant:0.4467948794395036
AP@sheep:0.7429324853127652
AP@sofa:0.7687370937413873
AP@train:0.824725421760546
AP@tvmonitor:0.7438172796180261
best MAP: 0.7528256433086052
"""