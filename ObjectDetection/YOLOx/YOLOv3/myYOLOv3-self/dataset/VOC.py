"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/15 14:09
"""
import cv2
import os
import torch
import json
import random
from PIL import Image
from lxml import etree
from configs.config import *
from utiles.encoder import Encoder
from utiles.iou import box_iou
from torch.utils.data import Dataset

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

class VOCDataSet(Dataset):
    """读取解析PASCAL VOC2007/2012数据集"""

    def __init__(
            self, voc_root,anchors, year="2012",
            transforms=None, train_set='train.txt',
            img_size = 416,S = (13,26,52),B = 3,num_classes = 20,
            is_train = True
    ):
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        # 增加容错能力
        if "VOCdevkit" in voc_root:
            self.root = os.path.join(voc_root, f"VOC{year}")
        else:
            self.root = os.path.join(voc_root, f"VOC{year}")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        txt_list = os.path.join(self.root, "ImageSets", "Main", train_set)

        with open(txt_list) as read:
            self.xml_list = [
                os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines() if len(line.strip()) > 0
            ]

        # read class_indict
        json_file = r"dataset/pascal_voc_classes.json"
        # json_file = r'/home/ff/YOLO/projects/yolov1/dataset/pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:
            self.class_dict = json.load(f)

        self.transforms = transforms
        self.img_size = img_size
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.anchors = anchors
        self.is_train = is_train
        self.encoder = Encoder(
            anchors=self.anchors,img_size=self.img_size,
            S = self.S,B = self.B,num_classes=self.num_classes
        )

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        height_width = [data_height, data_width]
        #read image and transform the image style
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        w,h = image.size

        assert "object" in data, "{} lack of object information.".format(xml_path)
        boxes = []
        labels = []
        for obj in data["object"]:
            # 将所有的gt box信息转换成相对值0-1之间
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]] - 1)
        boxes = torch.Tensor(boxes)
        labels = torch.LongTensor(labels)
        # Data augmentation while training.
        if self.is_train:
            image, boxes = self.random_flip(image, boxes)
            image, boxes, labels = self.random_crop(image, boxes, labels)

        image_t = image.resize((self.img_size, self.img_size))
        box_wh = torch.tensor([w,h,w,h],dtype=torch.float).expand_as(boxes)
        boxes /= box_wh

        if self.transforms is not None:
            image_t = self.transforms(image_t)
        loc_targets, cls_targets, boxes_targets = self.encoder.encoder(boxes,labels)
        for i in range(len(self.S)):
            loc_targets[i], cls_targets[i], boxes_targets[i] = torch.as_tensor(loc_targets[i]),\
                                                 torch.as_tensor(cls_targets[i]),torch.as_tensor(boxes_targets[i])
        return image_t, loc_targets,cls_targets,boxes_targets

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args：
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def random_flip(self, img, boxes):
        '''Randomly flip the image and adjust the bbox locations.

        For bbox (xmin, ymin, xmax, ymax), the flipped bbox is:
        (w-xmax, ymin, w-xmin, ymax).

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].

        Returns:
          img: (PIL.Image) randomly flipped image.
          boxes: (tensor) randomly flipped bbox locations, sized [#obj, 4].
        '''
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
        return img, boxes

    def random_crop(self, img, boxes, labels):
        '''Randomly crop the image and adjust the bbox locations.

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) bbox labels, sized [#obj,].

        Returns:
          img: (PIL.Image) cropped image.
          selected_boxes: (tensor) selected bbox locations.
          labels: (tensor) selected bbox labels.
        '''
        imw, imh = img.size
        while True:
            min_iou = random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
            if min_iou is None:
                return img, boxes, labels

            for _ in range(100):
                #随机裁剪的图形的[w,h]
                w = random.randrange(int(0.1*imw), imw)
                h = random.randrange(int(0.1*imh), imh)

                #判断裁剪的图像大小
                if h > 2*w or w > 2*h:
                    continue

                #开始裁剪的图像从哪个位置开始裁剪
                x = random.randrange(imw - w)
                y = random.randrange(imh - h)
                #定义裁剪的区域
                roi = torch.Tensor([[x, y, x+w, y+h]])

                #得到图像中box的中心坐标
                center = (boxes[:,:2] + boxes[:,2:]) / 2  # [N,2]
                #定义box的裁剪区域
                roi2 = roi.expand(len(center), 4)  # [N,4]
                #判断原图box的中心是否在裁剪区域之外
                mask = (center > roi2[:,:2]) & (center < roi2[:,2:])  # [N,2]
                #只有当box还在裁剪的区域中合理
                mask = mask[:,0] & mask[:,1]  #[N,]
                #如果裁剪之后的图像中已经没有包含的box了，则进行如下一轮的裁剪
                if not mask.any():
                    continue

                #得到裁剪之后的box
                selected_boxes = boxes.index_select(0, mask.nonzero(as_tuple=False).squeeze(1))

                #计算裁剪之后的box与裁剪之后图像的IOU，如果小于定义的程度，那么意味着box大部分很可能在裁剪的图像区域之外
                ious = box_iou(selected_boxes, roi)
                if ious.min() < min_iou:
                    continue

                #对图像进行裁剪
                img = img.crop((x, y, x+w, y+h))
                #得到裁剪之后的[x,y,w,h]
                selected_boxes[:,0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:,2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,3].add_(-y).clamp_(min=0, max=h)
                return img, selected_boxes, labels[mask]
    @staticmethod
    def collate_fn(batch):
        loc_targets,cls_targets,boxes = [],[],[]
        for _,loc_target,cls_target,box in batch:
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
            boxes.append(box)
        images = torch.stack([x[0] for x in batch],dim=0)
        return images,loc_targets,cls_targets,boxes
        # return torch.stack([x[0] for x in batch]), \
        #        torch.stack([x[1] for x in batch]), \
        #        torch.stack([x[2] for x in batch]), \
        #        [x[3] for x in batch]

if __name__ == '__main__':
    dataset = VOCDataSet(voc_root=r'E:\conda_3\PyCharm\Transer_Learning\PASCAL_VOC',anchors=ANCHORS)
    size = len(dataset)
    print(dataset[0])
    # per_class_num = {'aeroplane':0,'bicycle':0,'bird':0,'boat':0,'bottle':0,'bus':0,'car':0,'cat':0,'chair':0,
    #                  'cow':0,'diningtable':0,'dog':0,'horse':0,'motorbike':0,'person':0,'pottedplant':0,'sheep':0,
    #                  'sofa':0,'train':0,'tvmonitor':0}
    # for i in range(size):
    #     img,gt_map,labels = dataset[i]
    #     for j in labels:
    #         per_class_num[VOC_CLASSES[j - 1]] += 1
    #
    # for key,value in per_class_num.items():
    #     print('class: {} ---- number: {}'.format(key,value))
# import transforms
# from draw_box_utils import draw_objs
# from PIL import Image
# import json
# import matplotlib.pyplot as plt
# import torchvision.transforms as ts
# import random
#
# # read class_indict
# category_index = {}
# try:
#     json_file = open('./pascal_voc_classes.json', 'r')
#     class_dict = json.load(json_file)
#     category_index = {str(v): str(k) for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)
#
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
#
# # load train data set
# train_data_set = VOCDataSet(os.getcwd(), "2012", data_transform["train"], "train.txt")
# print(len(train_data_set))
# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target = train_data_set[index]
#     img = ts.ToPILImage()(img)
#     plot_img = draw_objs(img,
#                          target["boxes"].numpy(),
#                          target["labels"].numpy(),
#                          np.ones(target["labels"].shape[0]),
#                          category_index=category_index,
#                          box_thresh=0.5,
#                          line_thickness=3,
#                          font='arial.ttf',
#                          font_size=20)
#     plt.imshow(plot_img)
#     plt.show()
