"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/8/2 19:16
"""

import os
import torchvision
import torch
import random
from PIL import Image
from utils.utils import NMS, parse_out_MN
from utils.viz_result import viz_result
from net.densebox import DenseBox
import shutil
import re


def test():
    root = './img'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenseBox()
    weight_path = r'weights/model_49.pth'
    checkpoint = torch.load(weight_path,map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    for img_name in os.listdir(root):
        img_path = os.path.join(root,img_name)
        img = Image.open(img_path)
        W, H = img.size
        # ---------------- transform
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(H, W)),
            torchvision.transforms.CenterCrop(size=(H, W)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        img = transform(img)
        img = img.view(1, 3, H, W)

        # inference on gpu side
        img = img.to(device)
        score_out, loc_out = model.forward(img)

        # parse output on cpu side
        dets = parse_out_MN(score_map=score_out.cpu(),loc_map=loc_out.cpu(),
                            M=H,N=W,K=1)
        # do non-maximum suppression
        keep = NMS(dets=dets, nms_thresh=0.3)
        dets = dets[keep]
        # visualize final results
        viz_result(img_path=img_path,dets=dets,dst_root=r'runs')

if __name__ == "__main__":
    model_path = 'weights/model_49.pth'
    test()