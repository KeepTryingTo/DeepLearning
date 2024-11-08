import torch
from configs import config as cfg
from itertools import product as product
import math


class PriorBox(object):
    def __init__(self,device):
        super(PriorBox, self).__init__()
        self.img_size = cfg.img_size
        self.num_priors = len(cfg.aspect_ratios)
        self.variance = cfg.variance
        self.feature_maps = cfg.feature_maps
        self.min_sizes = cfg.min_sizes
        self.max_sizes = cfg.max_sizes
        self.steps = cfg.steps
        self.aspect_ratios = cfg.aspect_ratios
        self.clip = True
        self.device = device

    def forward(self):
        coords = []
        for k, f in enumerate(self.feature_maps):
            for i,j in product(range(f), repeat=2):
                f_k = float(self.img_size)/self.steps[k]
                #TODO 遍历方式是从左往右，然后依次往下，i遍历的是高度，j遍历的是宽度
                cy = (i+0.5)/f_k
                cx = (j+0.5)/f_k
                print('i = {}  j = {}'.format(i,j))
                # TODO aspect ratio = 1
                h_w_0 = float(self.min_sizes[k])/self.img_size

                coords += [cy, cx, h_w_0, h_w_0]
                h_w_1 = math.sqrt(h_w_0*(float(self.max_sizes[k])/self.img_size))
                coords += [cy, cx, h_w_1, h_w_1]
                # TODO other aspect ratios
                for a_r in self.aspect_ratios[k]:
                    h = float(h_w_0)/math.sqrt(a_r)
                    w = float(h_w_0)*math.sqrt(a_r)
                    coords += [cy,cx,h,w]
                    coords += [cy,cx,w,h]
        output = torch.Tensor(coords).view(-1,4)  # [8732,4]
        if self.clip:
            output.clamp(min=0., max=1.)
        output = output.to(self.device)
        return output #[cy, cx, h, w]

if __name__ == '__main__':
    p = PriorBox('cpu')
    p.forward()
