"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/15 16:14
"""

# target = {('1','2'):[[1,2,3,4],[4,5,6,7]]}
# for (k1,k2) in target:
#     print(k1,k2)
#
# x = [1,2,3,4,5,6,7,8]
# print(x[::2])


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
        return (x1,y1,x2,y2)

    def __getitem__(self, index):
        img=self.dataset[index]
        # imgTo=Image.open(img)
        imgTo = cv2.imread(img)
        H, W, C = imgTo.shape
        boxes = []
        for box in self.positions[index]:
            x1,y1,x2,y2 = box
            x1,y1,x2,y2 = x1 / W,y1 / H,x2 / W,y2 / H
            boxes.append([x1,y1,x2,y2])

        return boxes
    def __len__(self):
        return len(self.dataset)

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
    #计算训练集中所有box的(xmin,ymin,xmax,ymax)的平均值
    m_x1,m_y1,m_x2,m_y2 = 0.,0.,0.,0.
    for i in range(len(mydataset)):
        box = mydataset[i]
        for j in range(len(box)):
            m_x1 += box[j][0]
            m_y1 += box[j][1]
            m_x2 += box[j][2]
            m_y2 += box[j][3]
    size = mydataset.__len__()
    m_x1 = m_x1 / size
    m_y1 = m_y1 / size
    m_x2 = m_x2 / size
    m_y2 = m_y2 / size
    #TODO 0.39919 0.27900 0.62485 0.58801
    #TODO 计算平均值主要是为了查看模型在学习小的样本过程中是否过拟合；确实，当训练的样本数比较少，并且图像中的人物位置比较大部分固定在一个
    #TODO 范围之内时，很容易过拟合
    print('[x1,y1,x2,y2]: {:.5f} {:.5f} {:.5f} {:.5f}'.format(m_x1,m_y1,m_x2,m_y2))