

import os
import argparse
from train import _train
from infer import _infer
from data import *
from utils import *
from models.PFPNetR import build_pfp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    if not os.path.exists(args.save_folder + args.model_name + '/'):
        os.mkdir(args.save_folder + args.model_name + '/')

    cfg = data_configs[args.dataset][args.input_size]

    model = build_pfp(args.mode,
                      cfg['min_dim'],
                      cfg['num_classes'],
                      device=device).to(device)
    VOC_ROOT = r'D:\conda3\Transfer_Learning\PASCAL_VOC\VOCdevkit'
    if args.mode == 'train':

        if args.dataset == 'VOC':

            dataset = [VOCDetection(root=VOC_ROOT, mode=args.mode,
                                    transform=SSDAugmentation(size=cfg['min_dim'])),
                       VOCDetection(root=VOC_ROOT, mode='test',
                                    image_sets=[('2007', 'test')],
                                    transform=BaseTransform(args.input_size,
                                                            mean=(104, 117, 123)))
                       ]
        elif args.dataset == 'COCO':
            pass
        else:
            raise KeyError

        print('Start train...')
        _train(model, args, cfg, dataset,device)

    elif args.mode == 'test':

        if args.dataset == 'VOC':
            dataset = VOCDetection(root=VOC_ROOT, mode=args.mode,
                                   image_sets=[('2007', 'test')],
                                   transform=BaseTransform(args.input_size,
                                                           mean=(104, 117, 123)))
        elif args.dataset == 'COCO':
            pass
        else:
            raise KeyError

        print('Start inference')
        _infer(model, args, cfg, dataset,device)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--model_name',default='PFPNetR', type=str)

    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'], type=str, help='VOC or COCO')
    parser.add_argument('--input_size', default='320', choices=['320', '512'], type=str)
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')

    # Train args
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--iter_size', default=2, type=int, help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--basenet', default=r'D:\conda3\Transfer_Learning\ObjectDetect\RFBNet-master\weights\vgg16_reducedfc.pth',
                                    help='Pretrained base model')
    parser.add_argument('--save_folder', default='weights/', help='Directory for saving checkpoint models')
    parser.add_argument('--save_freq', default=5000, type=int)
    parser.add_argument('--valid_freq', default=5000, type=int)
    parser.add_argument('--print_freq', default=100, type=int)

    # Test args
    parser.add_argument('--test_model', default='./weights/PFPNetR320.pth', type=str, help='model for inference')
    parser.add_argument('--eval_folder', default='./eval/', type=str, help='Directory for the output of evaluation')
    parser.add_argument('--confidence_threshold', default=0.01, type=float, help='Detection confidence threshold')
    parser.add_argument('--top_k', default=200, type=int, help='Further restrict the number of predictions to parse')
    args = parser.parse_args()

    print(args)
    main(args)
