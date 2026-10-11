import numpy as np
import random

class SmallObjectAugmentation(object):
    def __init__(self, thresh=64*64, prob=0.5, copy_times=3, epochs=30, all_objects=False, one_object=False):
        """
        sample = {'img':img, 'annot':annots}
        img = [height, width, 3]
        annot = [xmin, ymin, xmax, ymax, label]
        thresh：the detection threshold of the small object. If annot_h * annot_w < thresh, the object is small
        prob: the prob to do small object augmentation
        epochs: the epochs to do
        """
        self.thresh = thresh
        self.prob = prob
        self.copy_times = copy_times
        self.epochs = epochs
        self.all_objects = all_objects
        self.one_object = one_object
        #TODO 如果是 复制-粘贴 一张图像中的所有小目标的话，就复制一次，如果是 复制-粘贴 多个目标（不是所有）的话就 复制-粘贴 多次
        if self.all_objects or self.one_object:
            self.copy_times = 1

    def issmallobject(self, h, w):
        if h * w <= self.thresh:
            return True
        else:
            return False

    def compute_overlap(self, annot_a, annot_b):
        #TODO 判断 复制-粘贴 的物体和已有的物体是否存在重叠的情况
        if annot_a is None: return False
        left_max = max(annot_a[0], annot_b[0])
        top_max = max(annot_a[1], annot_b[1])
        right_min = min(annot_a[2], annot_b[2])
        bottom_min = min(annot_a[3], annot_b[3])
        inter = max(0, (right_min-left_max)) * max(0, (bottom_min-top_max))
        if inter != 0:
            return True
        else:
            return False

    def donot_overlap(self, new_annot, annots):
        #TODO 只有当当前 复制-粘贴 的物体和图像中所有其他物体不存在重叠的时候才执行操作
        for annot in annots:
            if self.compute_overlap(new_annot, annot): return False
        return True

    def create_copy_annot(self, h, w, annot, annots):
        #TODO 当前图像中存在的某个物体高宽
        annot = annot.astype(int)
        annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
        #TODO 采样epochs次
        for epoch in range(self.epochs):
            #TODO 随机的获取物体的 复制-粘贴 中心
            random_x, random_y = np.random.randint(int(annot_w / 2), int(w - annot_w / 2)), \
                                 np.random.randint(int(annot_h / 2), int(h - annot_h / 2))
            xmin, ymin = random_x - annot_w / 2, random_y - annot_h / 2
            xmax, ymax = xmin + annot_w, ymin + annot_h
            if xmin < 0 or xmax > w or ymin < 0 or ymax > h:
                continue
            new_annot = np.array([xmin, ymin, xmax, ymax, annot[4]]).astype(int)
            #TODO 判断 复制-粘贴 的物体是否和已有的物体重叠
            if self.donot_overlap(new_annot, annots) is False:
                continue

            return new_annot
        return None

    def add_patch_in_img(self, annot, copy_annot, image):
        copy_annot = copy_annot.astype(int)
        #TODO 根据 复制-粘贴 的物体坐标位置 粘贴到图像中的指定区域
        image[annot[1]:annot[3], annot[0]:annot[2], :] = image[copy_annot[1]:copy_annot[3], copy_annot[0]:copy_annot[2], :]
        return image

    def __call__(self, sample):
        #TODO 判断当前是 复制-粘贴 一张图像中所有物体还是某一个物体
        if self.all_objects and self.one_object: return sample
        #TODO 计算随机 复制-粘贴 的概率
        if np.random.rand() > self.prob: return sample

        img, annots = sample['img'], sample['annot']
        h, w= img.shape[0], img.shape[1]

        #TODO 从一张图像中选择小目标
        small_object_list = list()
        for idx in range(annots.shape[0]):
            annot = annots[idx]
            annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
            #TODO 判断当前定位物体是否属于一个小目标
            if self.issmallobject(annot_h, annot_w):
                small_object_list.append(idx)

        l = len(small_object_list)
        # TODO 没有小目标就直接返回 No Small Object
        if l == 0: return sample

        # Refine the copy_object by the given policy
        # TODO Policy 2: 随机选择小目标样本
        copy_object_num = np.random.randint(0, l)
        # TODO Policy 3: 判断是 复制-粘贴 所有目标还是一个物体
        if self.all_objects:
            copy_object_num = l
        # Policy 1:
        if self.one_object:
            copy_object_num = 1
        #TODO 随机的从小目标列表中选择多个小目标
        random_list = random.sample(range(l), copy_object_num)
        annot_idx_of_small_object = [small_object_list[idx] for idx in random_list]
        #TODO 根据选择的小目标获得对应标注信息
        select_annots = annots[annot_idx_of_small_object, :]
        annots = annots.tolist()
        for idx in range(copy_object_num):
            annot = select_annots[idx]
            annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]

            if self.issmallobject(annot_h, annot_w) is False: continue

            for i in range(self.copy_times):
                new_annot = self.create_copy_annot(h, w, annot, annots,)
                if new_annot is not None:
                    img = self.add_patch_in_img(new_annot, annot, img)
                    annots.append(new_annot)
        #TODO 将经过 复制-粘贴 的图像返回
        return {'img': img, 'annot': np.array(annots)}

