"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/23 20:45
"""

import os
import torch
from lxml import etree
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class VOCDataSet(Dataset):
    """读取解析PASCAL VOC2007/2012数据集"""
    def __init__(
            self, voc_root, year="2012",train_set='train.txt',
    ):
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        # 增加容错能力
        if "VOCdevkit" in voc_root:
            self.root = os.path.join(voc_root, f"VOC{year}")
        else:
            self.root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        txt_list = os.path.join(self.root, "ImageSets", "Main", train_set)

        with open(txt_list) as read:
            self.xml_list = [
                os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines() if len(line.strip()) > 0
            ]

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
        assert "object" in data, "{} lack of object information.".format(xml_path)
        boxes = []
        for obj in data["object"]:
            # 将所有的gt box信息转换成相对值0-1之间
            xmin = int(float(obj["bndbox"]["xmin"])) / data_width
            xmax = int(float(obj["bndbox"]["xmax"])) / data_width
            ymin = int(float(obj["bndbox"]["ymin"])) / data_height
            ymax = int(float(obj["bndbox"]["ymax"])) / data_height
            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue
            boxes.append([xmin, ymin, xmax, ymax])
        return boxes

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
"""
注意：这里的iou和我们前面计算预测和标注框的iou意义上是不相同的；
这里的iou主要是指两个框的大小相近程度，不是两个框的重叠程度
"""
def iou(box,cluster):
    # x中存放的是随机选中的k个框分别和box[i]最小宽度
    x = np.minimum(cluster[:, 0], box[0])
    ##x中存放的是随机选中的k个框分别和box[i]最小高度
    y = np.minimum(cluster[:, 1], box[1])
    # 计算k个框和box[i]各个之间的IOU值
    intersection = x * y
    area1 = box[0] * box[1]
    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection + 1e-6)
    return iou

def avg_iou(boxes, cluster):
    return np.mean([
        np.max(
            iou(boxes[i], cluster)
        ) for i in range(boxes.shape[0])
    ])

def kmeans(boxes,k = 5):
    #聚类的box个数
    num_boxes = np.shape(boxes)[0]
    #衡量每一个box和k个box簇之间大小的相近程度
    distance = np.zeros(shape=(num_boxes,k))
    #初始化包含num_boxes的一维数组，表示当前的聚类和前一步的聚类结果是否相同
    pre_cluster = np.zeros(shape=(num_boxes,))
    #随机从boxes中随着k个box作为聚类中心,replace表示不可取相同的数字
    cluster = boxes[np.random.choice(a = num_boxes,size=k,replace=False)]
    iteration = 0
    while True:
        #计算boxes中的每个box与其cluster之间的box大小相近程度
        for i in range(num_boxes):
            distance[i] = 1 - iou(boxes[i],cluster)
        #根据计算的distance，找到每一个box和cluster之间distance最小的那个聚类中心，然后将该box分配该相应的聚类中心
        #在列的方向求解,得到和相应cluster最小的distance索引号
        min_dist = np.argmin(distance,axis=1)
        """
        np.all(np.array)   对矩阵所有元素做与操作，所有为True则返回True
        np.any(np.array)   对矩阵所有元素做或运算，存在True则返回True
        """
        if (pre_cluster == min_dist).all():
            break
        #根据计算的min_dist来为每个box分配到相应的簇中
        for i in range(k):
            #在行的方向上求解;median用于计算给定数据集或数组的中位数。其实就是在求解一个聚类的中心
            cluster[i] = np.median(boxes[min_dist == i],axis=0)
        pre_cluster = min_dist
        iteration += 1
        if iteration % k == 0:
            print('iterate: {:d}  avg_iou: {:.2f}'.format(iteration,avg_iou(boxes,cluster)))
    return cluster,min_dist

def drawAndSave(cluster,boxes,min_dist):
    for j in range(cluster.shape[0]):
        #将所有的点根据聚类的中心点进行绘制
        plt.scatter(boxes[min_dist == j][:,0], boxes[min_dist == j][:,1])
        #绘制聚类中心
        plt.scatter(cluster[j][0], cluster[j][1], marker='x', c='black')
    plt.savefig("kmeans_for_anchors.jpg")
    plt.show()
    print('Save kmeans_for_anchors.jpg in root dir.')

    #根据求解每一个anchor的框面积对其进行从小到大排序np.argsort，
    # 得到其在原数组中对应的索引，然后得到排序的结果
    cluster = cluster[np.argsort(cluster[:, 0] * cluster[:, 1])]
    print('avg_ratio:{:.2f}'.format(avg_iou(boxes, cluster)))

    f = open("generate_voc2012_anchors.txt", 'w')
    #将结果保存到文件中
    for i in range(cluster.shape[0]):
        if i == 0:
            x_y = "%d,%d" % (cluster[i][0], cluster[i][1])
        else:
            x_y = ", %d,%d" % (cluster[i][0], cluster[i][1])
        f.write(x_y)
    f.close()

if __name__ == '__main__':
    voc_root = r'E:\conda_3\PyCharm\Transer_Learning\PASCAL_VOC'
    dataset = VOCDataSet(voc_root=voc_root,year='2012',train_set='trainval.txt')
    boxes = []
    for i in range(dataset.__len__()):
        for box in dataset[i]:
            xmin,ymin,xmax,ymax = box
            boxes.append([xmax - xmin,ymax-ymin])
    boxes = np.array(boxes)
    cluster,min_dist = kmeans(boxes,k=5)
    img_size = (416,416)
    # 将所有的[0-1]之间的值转换为相对于416 x 416图像大小的值
    boxes = boxes * np.array([img_size[1], img_size[0]])
    ##将所有的[0-1]之间的值转换为相对于416 x 416图像大小的值
    cluster = cluster * np.array([img_size[1], img_size[0]])
    drawAndSave(cluster,boxes,min_dist)
    pass