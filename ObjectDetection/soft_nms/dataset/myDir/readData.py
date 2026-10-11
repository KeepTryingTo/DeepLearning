import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(self,rootDir,positionDir):
        self.data_dir=os.listdir(rootDir)
        #文件按序号列出
        self.data_dir.sort(key=lambda x:int(x.split('.')[0]))
        self.dataset=[]
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
            tuplelist=self.Xmin_Xmax_Ymin_Ymax(txt_path=txt_path)
            self.sorts.append([tuplelist[0]])
            self.positions.append([[tuplelist[1],tuplelist[2],tuplelist[3],tuplelist[4]]])
        print('data.Size: {}'.format(len(self.positions)))
        print('sort.Size: {}'.format(len(self.sorts)))

    def Xmin_Xmax_Ymin_Ymax(self,txt_path):
        """
        :param img_path: 图片文件的路径
        :param txt_path: 坐标文件的路径
        :return:
        """
        # 读取TXT文件 中的中心坐标和框大小
        with open(txt_path, "r") as fp:
            # 以空格划分
            contline = fp.readline().split(' ')
            # contline : class  x_center y_center width height
        # 返回：类别,xleft,yleft,xright,yright
        x1 = np.float64(contline[1])
        x2 = np.float64(contline[2])
        x3 = np.float64(contline[3])
        x4 = np.float64(contline[4])
        label=int(contline[0])
        return (label, x1, x3, x2, x4)

    def __getitem__(self, index):
        img=self.dataset[index]
        # imgTo=Image.open(img)
        imgTo=cv2.imread(img)/255
        #这里需要变换通道(H,W,C)=>(C,H,W)
        #方式一：
        newImg=np.transpose(imgTo,(2,0,1))
        #转换为tensor类型，这里如果使用torch.tensor(newImg)转换图像类型的话，
        # 后面在输入网络时就会出错：RuntimeError: expected scalar type Double but found Float
        newImg=torch.Tensor(newImg)

        #方式二：
        # newImg=torch.Tensor(imgTo).permute(2,0,1)
        PosTensor=torch.tensor(self.positions[index])
        LabTensor=torch.tensor(self.sorts[index])
        return newImg,PosTensor,LabTensor

    def __len__(self):
        dataSize=len(self.data_dir)
        return dataSize

if __name__ == '__main__':
    mydataset=myDataset(rootDir='data/fasterRcnn/test/img',positionDir='data/fasterRcnn/test/txt')
    # mydataset=myDataset_1(rootDir='data/train/trainDataset',positionDir='data/XML/trainDataset')
    print(len(mydataset))
    img,position,sort=mydataset[220]
    print('img: {}'.format(img))
    print('imgsize: {}'.format(img.size))
    # img.show('img')
    print('img.shape: {}'.format(np.shape(img)))
    print('img.type: {}'.format(type(img)))
    print('position.type: {}'.format(type(position)))
    # position=torch.Tensor(position)
    print('position.shape: {}'.format(position.shape))
    # sort=torch.Tensor(sort)
    print('sort: {}'.format(sort))
    print('sort.shape: {}'.format(np.shape(sort)))
    print('sort.type: {}'.format(type(sort)))