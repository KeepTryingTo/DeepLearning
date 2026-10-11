"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/15 14:09
"""
import cv2
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree

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
            self, voc_root, year="2012",
            transforms=None, train_set='train.txt',
            img_size = 448,S = 7,B = 2,num_classes = 20
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

    def __len__(self):
        return len(self.xml_list)

    def x1y1x2y2Tocxcywh(self,boxes):
        x1,x2,y1,y2 = boxes
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return (cx,cy,w,h)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        gt_map = torch.zeros(size=(self.S, self.S, 5 * self.B + self.num_classes))
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        height_width = [data_height, data_width]
        #read image and transform the image style
        img_path = os.path.join(self.img_root, data["filename"])
        image = cv2.imread(img_path)
        image = cv2.resize(image,dsize=(self.img_size,self.img_size))
        image_t = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_t = Image.fromarray(image_t)
        # if image.format != "JPEG":
        #     raise ValueError("Image '{}' format not JPEG".format(img_path))

        assert "object" in data, "{} lack of object information.".format(xml_path)
        boxes = []
        labels = []
        for obj in data["object"]:
            # 将所有的gt box信息转换成相对值0-1之间
            xmin = float(obj["bndbox"]["xmin"]) / data_width
            xmax = float(obj["bndbox"]["xmax"]) / data_width
            ymin = float(obj["bndbox"]["ymin"]) / data_height
            ymax = float(obj["bndbox"]["ymax"]) / data_height

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue
            boxes.append([xmin, xmax, ymin, ymax])
            labels.append(self.class_dict[obj["name"]])
        cell_size = 1 / self.S
        for i,(label,box) in enumerate(zip(labels,boxes)):
            class_label = int(label)
            #box的值已经进行了归一化处理
            cx,cy,w,h = self.x1y1x2y2Tocxcywh(boxes=box)
            ###########################  test the cx,cy,w,h #############################
            # x1,y1,x2,y2 = int((cx - w / 2) * self.img_size),\
            #               int((cy - h / 2) * self.img_size),\
            #               int((cx + w / 2) * self.img_size),\
            #               int((cy + h / 2) * self.img_size)
            # cv2.rectangle(img=image,pt1=(x1,y1),pt2=(x2,y2),color=(0,255,255),thickness=1)
            # cv2.imshow('img',image)
            # cv2.waitKey(0)
            #############################################################################
            # 计算网格中心点的左上角坐标以及相对于左上角的中心坐标
            i, j = int(cx / cell_size), int(cy / cell_size)
            #将其(j,i)转换为[0,1]之间的左上角
            x, y = i * cell_size,j * cell_size
            x_cell, y_cell = (cx - x) / cell_size, (cy - y) / cell_size
            #############################################################################
            # cx = x_cell * cell_size + x
            # cy = y_cell * cell_size + y
            # x1,y1,x2,y2 = int((cx - w / 2) * self.img_size),\
            #               int((cy - h / 2) * self.img_size),\
            #               int((cx + w / 2) * self.img_size),\
            #               int((cy + h / 2) * self.img_size)
            # cv2.rectangle(img=image,pt1=(x1,y1),pt2=(x2,y2),color=(0,255,255),thickness=1)
            # cv2.imshow('img',image)
            # cv2.waitKey(0)
            #############################################################################
            # 对框的高宽进行缩放
            # width_cell,height_cell = [
            #     w * self.S,
            #     h * self.S
            # ]
            width_cell, height_cell = [
                w, h
            ]
            if gt_map[j,i, class_label - 1] == 0:
                # 得到变换之后的坐标
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                # coordinate
                gt_map[j,i, self.num_classes:self.num_classes + 4] = box_coordinates
                gt_map[j,i, self.num_classes + 5:self.num_classes + 9] = box_coordinates
                # confidence
                gt_map[j,i, self.num_classes + 4] = 1
                gt_map[j,i, self.num_classes + 9] = 1
                # class label
                gt_map[j,i, class_label - 1] = 1
        if self.transforms is not None:
            image_t = self.transforms(image_t)
        return image_t, gt_map

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

    @staticmethod
    def collate_fn(batch):
        images, targets = tuple(zip(*batch))
        return images, targets

if __name__ == '__main__':
    dataset = VOCDataSet(voc_root=r'E:\conda_3\PyCharm\Transer_Learning\PASCAL_VOC')
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
