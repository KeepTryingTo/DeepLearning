"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/21-9:12
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
import thop

def compute_flops(model,img_size,device):
    x = torch.randn(1, 3, img_size[0], img_size[1]).to(device)
    from thop import profile
    MACs, params = profile(model, inputs=(x,), verbose=False)
    MACs, FLOPs, params = thop.clever_format([MACs, MACs * 2, params], "%.3f")
    print('MACs: {}'.format(MACs))
    print('FLOPs: {}'.format(FLOPs))
    print('params: {}'.format(params))