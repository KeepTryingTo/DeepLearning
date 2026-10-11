"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/24 20:04
"""

import torch
from torch import nn

def meshgrid_xy(fmsize):
    x = torch.arange(start=0,end=fmsize,step = 1)
    y = torch.arange(start=0,end=fmsize,step = 1)
    xx,yy = torch.meshgrid(x,y)
    xx,yy = xx.unsqueeze(dim = 2),yy.unsqueeze(dim = 2)
    #[fmsize,fmsize,2]
    xy = torch.stack([xx,yy],dim = 2)
    return xy

if __name__ == '__main__':
    print(meshgrid_xy(3))
    x = 3
    a = torch.arange(0, x)
    xx = a.view(-1, 1).repeat(1, x).view(-1, 1)
    yy = a.repeat(x, 1).view(-1, 1)
    xy = torch.cat([yy, xx], 1)
    xy = xy.view(x, x, 1, 2).expand(x, x, 5, 2)
    print(xy)
    pass
