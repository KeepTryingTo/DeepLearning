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


def test(src_root, dst_root, resume=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.isdir(src_root):
        print('=> [Err]: invalid root.')
        return

    # ---------------- network
    net = DenseBox()
    # print('=> net:\n', net)
    # ---------------- whether to resume from checkpoint
    assert resume is not None, 'must have pt file!!! '
    if resume is not None:
        if os.path.isfile(resume):
            net.load_state_dict(torch.load(resume))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Note]: invalid resume path @ %s, resume failed.' % resume)

    # ---------------- image
    imgs_path = [src_root + '/' + x for x in os.listdir(src_root)]

    # ---------------- clear dst root
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(dst_root)

    # ---------------- inference mode
    net.eval()
    net.to(device)

    # ---------------- inference
    for img_path in imgs_path:
        if os.path.isfile(img_path):
            # load image data and transform image
            img_name = os.path.split(img_path)[1]
            match = re.match('.*_label_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)', img_name)
            leftup_x = float(match.group(1))
            leftup_y = float(match.group(2))
            rightdown_x = float(match.group(3))
            rightdown_y = float(match.group(4))
            print('img: ', img_path)
            print('gt: [', leftup_x, ' ', leftup_y, '], [', rightdown_x, ' ', rightdown_y, ']')

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
            score_out, loc_out = net.forward(img)

            # parse output on cpu side
            dets = parse_out_MN(score_map=score_out.cpu(),
                                loc_map=loc_out.cpu(),
                                M=H,
                                N=W,
                                K=1)

            # do non-maximum suppression
            keep = NMS(dets=dets, nms_thresh=0.3)
            dets = dets[keep]

            # visualize final results
            viz_result(img_path=img_path,
                       dets=dets,
                       dst_root=dst_root)

            print('')


if __name__ == "__main__":
    model_path = 'weights/model_17.pth'
    src_root = './img'
    dst_root = './output'
    test(src_root, dst_root, resume=model_path)