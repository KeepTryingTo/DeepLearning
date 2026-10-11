"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/10 22:49
"""

import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(
            self,rootDir,positionDir,S = 7,B = 2,
            num_classes = 2,img_size = 448,
            transforms = None
    ):
        self.data_dir=os.listdir(rootDir)
        #文件按序号列出
        self.data_dir.sort(key=lambda x:int(x.split('.')[0]))
        self.dataset=[]
        self.img_size = img_size
        # assert transforms is not None,"the transforms can't empty!"
        self.transforms = transforms
        for img_ in self.data_dir:
            img_path=os.path.join(rootDir,img_)
            self.dataset.append(img_path)

        #获取.txt文件中的类别和坐标位置
        self.positions=[]
        self.sorts=[]
        self.poSortsDir=os.listdir(positionDir)
        self.poSortsDir.sort(key=lambda x:int(x.split('.')[0]))
        for txt_ in self.poSortsDir:
            txt_path=os.path.join(positionDir,txt_)
            tuplelists=self.Xmin_Xmax_Ymin_Ymax(txt_path=txt_path)
            labels = []
            boxes = []
            for tuplelist in tuplelists:
                labels.append(tuplelist[0])
                box = self.x1y1x2y2Tocxcywh(tuplelist)
                boxes.append(box)
            self.sorts.append(labels)
            self.positions.append(boxes)
        print('data.Size: {}'.format(len(self.positions)))
        print('sort.Size: {}'.format(len(self.sorts)))
        #初始化一个维度为[7,7,2*B + num_classes]张量
        self.S = S
        self.B = B
        self.num_classes = num_classes

    def Xmin_Xmax_Ymin_Ymax(self,txt_path):
        """
        :param img_path: 图片文件的路径
        :param txt_path: 坐标文件的路径
        :return:
        """
        lab_boxes = []
        # 读取TXT文件 中的中心坐标和框大小
        with open(txt_path, "r") as fp:
            contains = fp.readlines()
            # contline : class  x_center y_center width height
        for contline in contains:
            contline = contline.split(' ')
            # 返回：类别,xmin,xmax,ymin,ymax
            xmin = np.float64(contline[1])
            xmax = np.float64(contline[2])
            ymin = np.float64(contline[3])
            ymax = np.float64(contline[4])
            label=int(contline[0])
            #label xmin xmax ymin ymax
            lab_box = (label,xmin,xmax,ymin,ymax)
            lab_boxes.append(lab_box)
        return lab_boxes

    def x1y1x2y2Tocxcywh(self,boxes):
        label,x1,x2,y1,y2 = boxes
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return (cx,cy,w,h)

    def resizeBoxes(self,boxes,org_w,org_h,new_w,new_h):
        w_ratio = new_w / org_w
        h_ratio = new_h / org_h
        boxes[0] = boxes[0] * w_ratio
        boxes[1] = boxes[1] * h_ratio
        boxes[2] = boxes[2] * w_ratio
        boxes[3] = boxes[3] * h_ratio

        return boxes

    def __getitem__(self, index):
        img=self.dataset[index]
        # imgTo=Image.open(img)
        imgTo=cv2.imread(img)
        H,W,C = imgTo.shape
        #定义一个用于存储gt的[7,7,num_classes + B * 5]矩阵
        gt_map = torch.zeros(size=(self.S, self.S, 5 * self.B + self.num_classes))
        #这里需要变换通道(H,W,C)=>(C,H,W)
        #方式一：
        imgTo = cv2.resize(imgTo,dsize=(self.img_size,self.img_size))
        newImg = cv2.cvtColor(imgTo,cv2.COLOR_BGR2RGB)
        t_newImg = None
        #同时对图像和坐标缩放至给定的图像大小，这样做也是便于后面数据进行打包
        if self.transforms:
            newImg = Image.fromarray(newImg)
            t_newImg = self.transforms(newImg)
        #由于对图像进行了缩放，相应的坐标也需要进行缩放
        #将图像和坐标缩放至
        #转换为tensor类型，这里如果使用torch.tensor(newImg)转换图像类型的话，
        # 后面在输入网络时就会出错：RuntimeError: expected scalar type Double but found Float
        # newImg=torch.Tensor(newImg)
        cell_size = 1 / self.S
        for class_label,box in zip(self.sorts[index],self.positions[index]):
            class_label = int(class_label)
            cx,cy,w,h = box
            ###########################  test the cx,cy,w,h #############################
            # x1,y1,x2,y2 = int(cx - w / 2),int(cy - h / 2),int(cx + w / 2),int(cy + h / 2)
            # cv2.rectangle(img=imgTo,pt1=(x1,y1),pt2=(x2,y2),color=(0,255,255),thickness=1)
            # cv2.imshow('img',imgTo)
            # cv2.waitKey(0)
            #############################################################################
            #对图像进行归一化，让后面的训练以及grid cell中的坐标更好处理
            cx,cy,w,h = cx / W,cy / H, w / W, h / H
            # 计算网格中心点的左上角坐标以及相对于左上角的中心坐标
            i, j = int(cx / cell_size), int(cy / cell_size)
            # 将其(j,i)转换为[0,1]之间的左上角
            x, y = i * cell_size, j * cell_size
            x_cell, y_cell = (cx - x) / cell_size, (cy - y) / cell_size
            #############################################################################
            # cx = x_cell * cell_size + x
            # cy = y_cell * cell_size + y
            # x1,y1,x2,y2 = int((cx - w / 2) * self.img_size),\
            #               int((cy - h / 2) * self.img_size),\
            #               int((cx + w / 2) * self.img_size),\
            #               int((cy + h / 2) * self.img_size)
            # cv2.rectangle(img=imgTo,pt1=(x1,y1),pt2=(x2,y2),color=(0,255,255),thickness=1)
            # cv2.imshow('img',imgTo)
            # cv2.waitKey(0)
            #############################################################################
            #对框的高宽进行缩放
            # width_cell,height_cell = [
            #     w * self.S,
            #     h * self.S
            # ]
            width_cell, height_cell = [
                w,h
            ]
            if gt_map[j,i,class_label] == 0:
                #得到变换之后的坐标
                box_coordinates = torch.tensor(
                    [x_cell,y_cell,width_cell,height_cell]
                )
                #coordinate
                gt_map[j,i,self.num_classes:self.num_classes + 4] = box_coordinates
                gt_map[j,i,self.num_classes + 5:self.num_classes + 9] = box_coordinates
                #confidence
                gt_map[j,i,self.num_classes + 4] = 1
                gt_map[j,i,self.num_classes + 9] = 1
                #class label
                gt_map[j,i,class_label] = 1

        #方式二：
        # newImg=torch.Tensor(imgTo).permute(2,0,1)
        # PosTensor=torch.tensor(self.positions[index])
        # LabTensor=torch.tensor(self.sorts[index])
        return t_newImg,gt_map

    def __len__(self):
        dataSize=len(self.data_dir)
        return dataSize

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift(self, bgr, boxes, labels):
        # 平移变换
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            # print(bgr.shape,shift_x,shift_y)
            # 原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(
                boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if (len(boxes_in) == 0):
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im

if __name__ == '__main__':
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        transforms.ToTensor()
    ])
    mydataset=myDataset(
        rootDir=r'E:\Data(D)\workspace\max\OK\train\person\train_img',
        positionDir=r'E:\Data(D)\workspace\max\OK\train\person\train_txt',
        transforms=transform
    )
    # mydataset=myDataset_1(rootDir='data/train/trainDataset',positionDir='data/XML/trainDataset')
    print(len(mydataset))
    img,gt_map=mydataset[22]
    print('img: {}'.format(img))
    print('imgsize: {}'.format(img.size))
    # img.show('img')
    print('img.shape: {}'.format(np.shape(img)))
    print('img.type: {}'.format(type(img)))
    print('gt_map: {}'.format(gt_map.size))