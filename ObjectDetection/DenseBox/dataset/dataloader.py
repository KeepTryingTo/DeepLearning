
'''
# -*- encoding: utf-8 -*-
# 文件    : dataloader.py
# 说明    : 数据集加载
# 时间    : 2022/06/28 17:12:38
# 作者    : Hito
# 版本    : 1.0
# 环境    : pytorch1.7
'''



import torch
from torch.utils import data
from torchvision import transforms as T
import os,re
from tqdm import tqdm
from PIL import Image


class LPPatch_Offline(data.Dataset):
    def __init__(self, root, transform=None,size=(240, 240)):

        if not os.path.isdir(root):
            print('=> [Err]: invalid root.')
            return

        # image size
        self.size = size

        # load image transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize(self.size),
                T.CenterCrop(self.size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # init images' path
        self.imgs_path = [root + '/' + x for x in os.listdir(root)[:2000]]
        if len(self.imgs_path) == 0:
            print('=> [Warning]: empty root.')
            return
        

    def precess(self, img_path):
        self.label_map = torch.zeros([5, 60, 60], dtype=torch.float32)
        # init lossmask map with 0
        self.mask_map = torch.zeros([1, 60, 60], dtype=torch.float32)
        # leftup_x, leftup_y, rightdown_x, right_down_y
        img_name = os.path.split(img_path)[1]
        """
        match()方法用于从字符串的开始位置进行匹配,如果起始位置匹配成功,则返回Match对象,否则返回None
        语法: re.match(pattern,string,[flags])
        pattern: 模式字符串   string:要匹配的字符串   flags:可选参数,比如re.I 不区分大小写
        """
        #match: 将根据图像文件名的label字符之后进行字符匹配
        match = re.match('.*_label_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)',
                            img_name)
        # 240×240 coordinate space, float
        #(leftup_x,leftup_y,rightdown_x,rightdown_y)表示边框的左上角和右下角坐标
        leftup_x = float(match.group(1))
        leftup_y = float(match.group(2))
        rightdown_x = float(match.group(3))
        rightdown_y = float(match.group(4))
        # turn coordinate to 60×60 coordinate space, float
        """
        由于原始图像经过网络之后经过了4倍的下采样，所以对其坐标进行4倍的缩放
        """
        leftup_x = float(leftup_x) / 4.0  # 240.0 = 4 * 60.0
        leftup_y = float(leftup_y) / 4.0  # 240.0 = 4 * 60.0
        rightdown_x = float(rightdown_x) / 4.0  # 240.0 = 4 * 60.0
        rightdown_y = float(rightdown_y) / 4.0  # 240.0 = 4 * 60.0
        # ------------------------- fill label map#score_map=self.label_map[0],
        self.init_score_map(leftup=(leftup_x, leftup_y),
                            rightdown=(rightdown_x, rightdown_y),ratio=0.3)
        # print(torch.nonzero(score_map))
        self.init_dist_map(leftup=(leftup_x, leftup_y),
                           rightdown=(rightdown_x, rightdown_y))
        # ------------------------- init loss mask map#mask_map=self.mask_map,
        self.init_mask_map(leftup=(leftup_x, leftup_y),
                           rightdown=(rightdown_x, rightdown_y))
        return self.label_map, self.mask_map

    def init_score_map(self,leftup,rightdown,ratio=0.3):
        # assert score_map.size == torch.Size([60, 60])
        #TODO 获得box的中心坐标以及高宽
        bbox_center_x = float(leftup[0] + rightdown[0]) * 0.5
        bbox_center_y = float(leftup[1] + rightdown[1]) * 0.5
        bbox_w = rightdown[0] - leftup[0]
        bbox_h = rightdown[1] - leftup[1]

        #TODO 根据指定的比率以及box的高宽得到微调之后的（xmin,ymin,xmax,ymax）
        org_x = int(bbox_center_x - float(ratio * bbox_w * 0.5) + 0.5)
        org_y = int(bbox_center_y - float(ratio * bbox_h * 0.5) + 0.5)
        end_x = int(float(org_x) + float(ratio * bbox_w) + 0.5)
        end_y = int(float(org_y) + float(ratio * bbox_h) + 0.5)
        #TODO 然后获得最终的label map
        self.label_map[0][org_y: end_y + 1, org_x: end_x + 1] = 1.0
        # verify...
        # print(torch.nonzero(score_map))

    def init_mask_map(self, leftup,  rightdown, ratio=0.3):
        # assert mask_map.size == torch.Size([1, 60, 60])
        bbox_center_x = float(leftup[0] + rightdown[0]) * 0.5
        bbox_center_y = float(leftup[1] + rightdown[1]) * 0.5
        bbox_w = rightdown[0] - leftup[0]
        bbox_h = rightdown[1] - leftup[1]
        org_x = int(bbox_center_x - float(ratio * bbox_w * 0.5) + 0.5)
        org_y = int(bbox_center_y - float(ratio * bbox_h * 0.5) + 0.5)
        end_x = int(float(org_x) + float(ratio * bbox_w) + 0.5)
        end_y = int(float(org_y) + float(ratio * bbox_h) + 0.5)
        self.mask_map[:, org_y: end_y + 1, org_x: end_x + 1] = 1.0


    def init_dist_map(self,leftup, rightdown):
        # assert dxt_map.size == torch.Size([60, 60])
        bbox_w = rightdown[0] - leftup[0]
        bbox_h = rightdown[1] - leftup[1]
        #TODO 根据label map对其进行遍历，计算每一个点和带有坐标的左上角和右下角之间的距离
        for y in range(self.label_map[1].size(0)):  # dim H
            for x in range(self.label_map[1].size(1)):  # dim W
                dist_xt = (float(x) - leftup[0]) / self.label_map[1].size(1)
                dist_yt = (float(y) - leftup[1]) / self.label_map[1].size(0)
                dist_xb = (float(x) - rightdown[0]) / self.label_map[1].size(1)
                dist_yb = (float(y) - rightdown[1]) / self.label_map[1].size(0)
                self.label_map[1][y, x] = dist_xt
                self.label_map[2][y, x] = dist_yt
                self.label_map[3][y, x] = dist_xb
                self.label_map[4][y, x] = dist_yb


    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx])
        # convert gray to RGB
        if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
            img = img.convert('RGB')
        # image data transformation
        if self.transform is not None:
            img = self.transform(img)
        label_map, mask_map = self.precess(self.imgs_path[idx])
        return img, label_map, mask_map
    def __len__(self):
        return len(self.imgs_path)

if __name__ == '__main__':
    dataset = LPPatch_Offline(root='./dataset',size=(240,240))
    print('dataset: {}'.format(dataset[0]))
    print(dataset.__len__())
    pass