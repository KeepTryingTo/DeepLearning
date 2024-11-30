"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/20 17:19
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
        json_file = r"../dataset/pascal_voc_classes.json"
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
        assert "object" in data, "{} lack of object information.".format(xml_path)

        return data

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
        Args：xml: xml tree obtained by parsing XML file contents using lxml.etree
        Returns:Python dictionary holding XML contents.
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

if __name__ == '__main__':
    dataset = VOCDataSet(voc_root=r'E:\conda_3\PyCharm\Transer_Learning\PASCAL_VOC')
    size = len(dataset)
    print(dataset[0])

