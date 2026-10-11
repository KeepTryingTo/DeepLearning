import os
import time
import argparse

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
import datasets.transforms as T
from torchvision import transforms
from models import build_model as build_yolos_model

from PIL import Image
import matplotlib.pyplot as plt
torch.set_grad_enabled(False)

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, scores,labels, boxes,conf_thresh,img_name):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    print('boxes.shape: {}'.format(np.shape(boxes)))
    for p,cls_idx, (xmin, ymin, xmax, ymax), c in zip(scores,labels, boxes.tolist(), COLORS * 100):
        if p > conf_thresh:
            ax.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            text = f'{CLASSES[cls_idx]}: {p:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(f'outputs/{img_name.split(".")[0]}.png')
    # plt.show()

def get_args_parser():
    parser = argparse.ArgumentParser('Set YOLOS', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--eval_size', default=800, type=int)

    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--use_checkpoint', action='store_true',
                        help='use checkpoint.checkpoint to save mem')
    # scheduler
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='warmupcos', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step", options:"step", "warmupcos"')
    ## step
    parser.add_argument('--lr_drop', default=100, type=int)
    ## warmupcosin
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # * model setting
    parser.add_argument("--det_token_num", default=100, type=int,
                        help="Number of det token in the deit backbone")
    parser.add_argument('--backbone_name', default='tiny', type=str,
                        help="Name of the deit backbone to use")
    parser.add_argument('--pre_trained',
                        default=r'./weights/deit_tiny_patch16_224.pth',
                        help="set imagenet pretrained model path if not train yolos from scatch")
    parser.add_argument('--init_pe_size', nargs='+',
                        type=int, default=(800, 1333),help="init pe size (h,w)")
    parser.add_argument('--mid_pe_size', nargs='+', type=int,
                        help="mid pe size (h,w)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients

    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco',type=str)
    parser.add_argument('--coco_path', type=str,
                        default=r'/home/ff/myProject/KGT/myProjects/myDataset/coco')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir',default=r'./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

if __name__ == '__main__':
    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    args = get_args_parser().parse_args()

    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = ['Argument', 'Value']
    for arg, value in vars(args).items():
        table.add_row([arg, value])
    print('-----------------------argument---------------------------')
    print(table)
    print('----------------------------------------------------------')

    model, criterion, postprocessors = build_yolos_model(args)
    state_dict = torch.load(
        f'/home/ff/myProject/KGT/myProjects/myProjects/YOLOS-main/runs/checkpoint.pth',
        map_location='cpu'
    )
    model.load_state_dict(state_dict['model'])
    model.eval()
    print('load model is done ...')
    print('clsses number: {}'.format(len(CLASSES)))

    # standard PyTorch mean-std input image normalization
    transform = transforms.Compose([
        transforms.Resize(size=(args.eval_size,args.eval_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225)
        )
    ])

    root = r'./images'
    img_list = os.listdir(root)
    for img_name in img_list:
        imgPath = os.path.join(root,img_name)
        im = Image.open(imgPath).convert('RGB')
        orig_target_sizes = torch.tensor(im.size,dtype=torch.float).unsqueeze(dim = 0)
        start_time = time.time()
        img_transform = transform(im).unsqueeze(dim = 0).to(device)
        outputs = model(img_transform)
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        # print('out_logits.shape: {}'.format(out_logits.size()))
        # print('out_bbox.shape: {}'.format(out_bbox.size()))
        # print('len(out_logits) = {}'.format(len(out_logits)))
        # print('len(orig_target_sizes) = {}'.format(len(orig_target_sizes)))
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        end_time = time.time()
        scores, boxes, labels = (results[0]['scores'],
                                 results[0]['boxes'],
                                 results[0]['labels'])
        print('boxes.shape: {}'.format(boxes.size()))
        plot_results(pil_img=im, scores=scores,
                     labels=labels, boxes=boxes,
                     conf_thresh=0.45,img_name=img_name)
        print('inference time: {}'.format(end_time - start_time))



