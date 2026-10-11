"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/15 14:09
"""

import torch
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    CLASSES_NAME = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(self, root_dir,year = '2012', resize_size=(800, 1024), split='trainval', use_difficult=False):

        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        # 增加容错能力
        if "VOCdevkit" in root_dir:
            self.root = os.path.join(root_dir, f"VOC{year}")
        else:
            self.root = os.path.join(root_dir, "VOCdevkit", f"VOC{year}")
        self.use_difficult = use_difficult
        self.imgset = split

        # 读取标注文件.XML
        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        # 读取XML对应的jpg图像
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        # 读取对应的Main文件下的对应图像的名称
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.imgset) as f:
            self.img_ids = f.readlines()
        self.img_ids = [x.strip() for x in self.img_ids]
        self.name2id = dict(zip(VOCDataset.CLASSES_NAME, range(len(VOCDataset.CLASSES_NAME))))
        # 缩放图像的大小
        self.resize_size = resize_size
        # 对图像进行归一化处理
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        print("INFO=====>voc dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def _read_img_rgb(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        # 读取图像数据集
        img = self._read_img_rgb(self._imgpath % img_id)
        # 根据XML文件获得图像的标注信息
        anno = ET.parse(self._annopath % img_id).getroot()
        boxes = []
        classes = []
        for obj in anno.iter("object"):
            # 判断该物体是否为容易检测
            difficult = int(obj.find("difficult").text) == 1
            # 如果是不容易检测物体，那么直接忽略
            if not self.use_difficult and difficult:
                continue
            _box = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            # 得到图像中标注物体的标注框信息
            box = [
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            ]
            TO_REMOVE = 1
            # 将标注框的长度都同时减1
            box = tuple(map(lambda x: x - TO_REMOVE, list(map(float, box))))
            boxes.append(box)
            # 得到物体的名称
            name = obj.find("name").text.lower().strip()
            # 保存物体的名称
            classes.append(self.name2id[name])

        # 将标注框的长度转换为numpy类型
        boxes = np.array(boxes, dtype=np.float32)
        # 对标注框的信息进行处理，当对图像进行缩放之后，相应的物体坐标信息也要进行缩放
        img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)
        # 将图像转换为tensor
        img = transforms.ToTensor()(img)
        # 将物体的坐标信息转换为numpy类型
        boxes = torch.from_numpy(boxes)
        # 将物体的类别转换为对应的tensor(long)
        classes = torch.LongTensor(classes)

        return img, boxes, classes

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side = input_ksize
        h, w, _ = image.shape

        # 获得物体的最段边长和最长边长
        smallest_side = min(w, h)
        largest_side = max(w, h)
        # 计算给定的最小边长和当前的物体的最小边长的比例
        scale = min_side / smallest_side
        # 如果该比例 X 最大边长 > 指定的最大边长，那么重新计算长度比
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        # 得到变换之后的宽度和长度
        nw, nh = int(scale * w), int(scale * h)
        # 对图像按照等比例进行缩放
        image_resized = cv2.resize(image, (nw, nh))
        # 将图像填充至32的倍数
        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32

        # 在图像周围进行0填充
        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            # 对图像进行变换的同时，相应的物体边框信息也要进行改变
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def collate_fn(self, data):
        # 将对应的图像数据，物体坐标以及类别信息打包
        imgs_list, boxes_list, classes_list = zip(*data)
        # 检查图像数和类别数以及坐标信息是否吻合
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        # 得到batchsize
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []

        # 得到一个batch中对应图像的高宽
        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        # 得到一个batch中对应的所有图像的最大高度和最大宽度
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        """
        torch.nn.functional.pad(input, pad, mode,value ) 
        input：四维或者五维的tensor Variabe
        pad：不同Tensor的填充方式
            1.四维Tensor：传入四元素tuple(pad_l, pad_r, pad_t, pad_b)，
            指的是（左填充，右填充，上填充，下填充），其数值代表填充次数
            2.六维Tensor：传入六元素tuple(pleft, pright, ptop, pbottom, pfront, pback)，
            指的是（左填充，右填充，上填充，下填充，前填充，后填充），其数值代表填充次数
        mode： ’constant‘, ‘reflect’ or ‘replicate’三种模式，指的是常量，反射，复制三种模式
        value：填充的数值，在"contant"模式下默认填充0，mode="reflect" or "replicate"时没有			
            value参数
        """
        for i in range(batch_size):
            img = imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)
                                 (torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]),
                                                                0, int(max_h - img.shape[1])), value=0.)))

        max_num = 0
        # 得到一个batch中所有图像中最大的物体数
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num:
                max_num = n
        # 对于一些图像中不足的物体数进行填充
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)

        return batch_imgs, batch_boxes, batch_classes


if __name__ == "__main__":
    import cv2

    dataset = VOCDataset("dataset/VOCdevkit/VOC2012", split='trainval')
    # for i in range(100):
    #     img,boxes,classes=dataset[i]
    #     img,boxes,classes=img.numpy().astype(np.uint8),boxes.numpy(),classes.numpy()
    #     img=np.transpose(img,(1,2,0))
    #     print(img.shape)
    #     print(boxes)
    #     print(classes)
    #     for box in boxes:
    #         pt1=(int(box[0]),int(box[1]))
    #         pt2=(int(box[2]),int(box[3]))
    #         img=cv2.rectangle(img,pt1,pt2,[0,255,0],3)
    #     cv2.imshow("test",img)
    #     if cv2.waitKey(0)==27:
    #         break
    imgs, boxes, classes = dataset.collate_fn([dataset[105], dataset[101], dataset[200]])
    print(boxes, classes, "\n", imgs.shape, boxes.shape, classes.shape, boxes.dtype, classes.dtype, imgs.dtype)
    for index, i in enumerate(imgs):
        i = i.numpy().astype(np.uint8)
        i = np.transpose(i, (1, 2, 0))
        i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
        print(i.shape, type(i))
        cv2.imwrite('assets/' + str(index) + ".jpg", i)
