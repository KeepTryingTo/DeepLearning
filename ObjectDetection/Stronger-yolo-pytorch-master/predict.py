"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/1/13-18:53
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import os
import time
import argparse
import numpy as np

import cv2
import torch
from torch import nn
from torchvision import transforms

from yacscfg import _C as cfg
from utils.nms_utils import torch_nms
from models.strongerv3kl import StrongerV3KL
from utils.visualize import visualize_boxes


def _postprocess(pred_bbox, test_input_size, org_img_shape,boxloss = 'KL', varvote = True):
    if boxloss == 'KL':
        pred_coor = pred_bbox[:, 0:4]
        pred_vari = pred_bbox[:, 4:8]
        pred_vari = torch.exp(pred_vari)
        pred_conf = pred_bbox[:, 8]
        pred_prob = pred_bbox[:, 9:]
    else:
        pred_coor = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]
    org_h, org_w = org_img_shape
    # TODO 计算输入到模型中的图像大小和原图的比
    resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
    # TODO 计算原图和输入图像之间的宽度和高度差
    dw = (test_input_size - resize_ratio * org_w) / 2
    dh = (test_input_size - resize_ratio * org_h) / 2
    # TODO 对预测的box进行缩放
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
    x1, y1, x2, y2 = torch.split(pred_coor, [1, 1, 1, 1], dim=1)
    # TODO 检查坐标的边界情况
    x1, y1 = torch.max(x1, torch.zeros_like(x1)), torch.max(y1, torch.zeros_like(y1))
    x2, y2 = torch.min(x2, torch.ones_like(x2) * (org_w - 1)), torch.min(y2, torch.ones_like(y2) * (org_h - 1))
    pred_coor = torch.cat([x1, y1, x2, y2], dim=-1)

    # ***********************
    if pred_prob.shape[-1] == 0:
        pred_prob = torch.ones((pred_prob.shape[0], 1)).cuda()
    # TODO ***********************
    scores = pred_conf.unsqueeze(-1) * pred_prob
    bboxes = torch.cat([pred_coor, scores], dim=-1)
    if boxloss == 'KL' and varvote:
        return bboxes, pred_vari
    else:
        return bboxes, None


def predictImage(
        model,root,
        img_size,catName,
        save_path = None,
        iou_threshold = 0.25
):
    s = time.time()
    model.eval()
    for imgName in os.listdir(root):
        start_time = time.time()

        img_path = os.path.join(root, imgName)
        image = cv2.imread(img_path)
        H,W,C = image.shape
        resize_ratio = min(1.0 * img_size / W, 1.0 * img_size / H)
        resize_w = int(resize_ratio * W)
        resize_h = int(resize_ratio * H)
        image_resized = cv2.resize(image, (resize_w, resize_h))
        image_paded = np.full((img_size, img_size, 3), 128.0)
        dw = int((img_size - resize_w) / 2)
        dh = int((img_size - resize_h) / 2)
        image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized
        img = image_paded / 255.0
        img = transforms.ToTensor()(img)
        img_t = img.unsqueeze(dim = 0).to(args.device).float()

        # print('type img_t: {}'.format(type(img_t)))

        with torch.no_grad():
            outputs = model(img_t)
        # TODO 进行预测box的解码操作
        print('outputs.shape: {}'.format(outputs.size()))
        bbox, bboxvari = _postprocess(outputs[0], img_size, org_img_shape = (H,W))
        nms_boxes, nms_scores, nms_labels = torch_nms(cfg.EVAL, bbox,
                                                      variance=bboxvari)
        print('boxes: {}'.format(nms_boxes.size()))
        print('scores: {}'.format(nms_scores.size()))
        print('labels: {}'.format(nms_labels.size()))

        nms_boxes = nms_boxes.cpu().numpy()
        nms_scores = nms_scores.cpu().numpy()
        nms_labels = nms_labels.cpu().numpy()
        image = visualize_boxes(
            image=image,
            boxes=nms_boxes,
            labels=nms_labels,
            probs=nms_scores,
            class_labels=catName,
            iou_threshold=iou_threshold,
            img_name=imgName
        )
        if save_path is not None:
            cv2.imwrite(os.path.join(save_path,imgName),image)
        else:
            cv2.imwrite('result.png', image)

        print('{} inference time is {}'.format(imgName,time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DEMO configuration")
    parser.add_argument(
        "--config-file",
        default='configs/strongerv3_all.yaml'
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
        default='cuda:0'
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)

    weight_path = r'./checkpoints/strongerv3_all/voc_checkpoint-best.pth'
    checkpoint = torch.load(weight_path, map_location='cpu')
    model = StrongerV3KL(cfg=cfg.MODEL)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    model.eval()

    catName = ["aeroplane","bicycle","bird","boat","bottle","bus","car",
        "cat","chair","cow","diningtable","dog","horse","motorbike",
        "person","pottedplant","sheep","sofa","train","tvmonitor"
    ]

    predictImage(
        model=model,
        root='./images',
        img_size=512,
        catName=catName,
        save_path='outputs',
        iou_threshold=0.25
    )







