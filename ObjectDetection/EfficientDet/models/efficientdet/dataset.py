import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        #读取.JSON文件(box，img_id，label_id)
        self.coco = COCO(
            os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json')
        )
        #得到所有图像的id
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label) 根据类别id得到类别name
        categories = self.coco.loadCats(self.coco.getCatIds())
        #根据类别id进行升序排序
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        #annot: [num_boxes,5]
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        #通过图像的id得到有关当前图像的所有信息：file_name，img_url，img_height，img_width，以及对应的id
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name']) #得到图像的路径
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #对图像进行归一化处理[0,1]
        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations 得到标注图像的box信息以及类别信息的id(指的是这个annotation的一个id)
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index],
            iscrowd=False
        )
        annotations = np.zeros((0, 5)) #保存box坐标以及label

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations 其中包含了segmentation和Bbox信息(坐标，类别id以及annotation下的id)
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them 去掉box中不符合要求的box
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox'] #得到Bbox坐标信息
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0) #https://blog.csdn.net/weixin_42216109/article/details/93889047

        # transform from [x, y, w, h] to [x1, y1, x2, y2] => [num_boxes,5 => [xmin,ymin,xmax,ymax]]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


def demoCOCODataset():
    root_dir = r'E:\conda_3\PyCharm\Transer_Learning\MSCOCO'
    imgs_path = os.path.join(root_dir,'coco','val2017')
    anno_path = os.path.join(root_dir,'coco','annotations','instances_val2017.json')

    samples = CocoDataset(
        root_dir = os.path.join(r'E:\conda_3\PyCharm\Transer_Learning\MSCOCO','coco'),
        set = 'val2017'
    )
    print('len: {}'.format(samples.__len__()))
    print('sample[0]: {}'.format(samples[0]))


if __name__ == '__main__':
    demoCOCODataset()
    pass

