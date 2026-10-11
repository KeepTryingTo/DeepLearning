import numpy as np
from PIL import Image


class GridMask(object):
    def __init__(self, use_h, use_w, rotate=1,
                 offset=False, ratio=0.5, mode=0, prob=1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, sample):
        img, annots = sample['img'], sample['annot']
        #todo 是否进行mask
        if np.random.rand() > self.prob:
            return sample
        #TODO 获得图像的高宽同时计算高宽两个中的最小值
        h = img.shape[0]
        w = img.shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        #TODO 在原图的基础上扩大一定的高宽
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        #TODO 选择单元长度
        d = np.random.randint(self.d1, self.d2)
        # d = self.d
        #        self.l = int(d*self.ratio+0.5)
        #TODO 默认ratio=0.5
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        #TODO mask大小
        mask = np.ones((hh, ww), np.float32)
        #TODO X和Y轴方向的mask偏移量
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            #TODO 根据d单位长度即可知道划分的单元数量
            for i in range(hh // d):
                s = d * i + st_h
                #TODO 边界处理
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        #TODO 旋转角度
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        # TODO mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]
        #TODO 默认mode=1
        if self.mode == 1:
            mask = 1 - mask
        #TODO 升维
        mask = np.expand_dims(mask.astype(float), axis=2)
        mask = np.tile(mask, [1, 1, 3])
        #TODO 是否进行偏移
        if self.offset:
            offset = float(2 * (np.random.rand(h, w) - 0.5))
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask
        return {'img': img, 'annot': annots}
