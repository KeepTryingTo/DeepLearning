"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/11/5-19:28
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import json
import os
import time
import xml.etree.ElementTree as ET
from tqdm import tqdm


def pascal_voc_to_coco(voc_dir, output_json_file):
    # TODO COCO格式的基础结构
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_dict = {}
    annotation_id = 1

    # TODO 定义类别（可以根据自己的 VOC 类别进行修改）
    """
        我们要知道COCO是80个类别，而VOC是20个类别，那么将VOC转换为COCO格式的时候，使用COCO
    的评估指标来评估VOC测试结果，必然在VOC中未出现的类别AP=0，那么最终计算的MAP结果应该是按照
    80个类别来计算的，因此最终的计算结果应该按照VOC的20个类别来，尽管COCO数据集为80个类别，但是
    我们会发现VOC中的一些类别并没有出现在COCO中
    """
    categories = [
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife",
        "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "TV", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]

    VOC_CLASSES = (
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')

    #TODO 将类别名称和索引对应起来（索引从1开始）
    for id, category in enumerate(VOC_CLASSES, start=1):
        category_dict[category] = id
        coco_output["categories"].append({
            "id": id,
            "name": category,
            "supercategory": "none",
        })

    #TODO 这里将PASCAL VOC07 test的标注文件XML转换为COCO支持的JSON文件格式
    #TODO 第一步：根据VOC07提供的test.TXT文件读取对应的XML文件
    test_txt = os.path.join(voc_dir,'ImageSets/Main','test.txt')
    with open(test_txt,'r',encoding='utf-8') as fp:
        lines = fp.readlines()
    xml_files = []
    for xmlName in lines:
        xml_files.append(
            xmlName.strip() + '.xml'
        )

    #TODO 根据读取的XML文件路径读取XML文件内容
    annotation_xml = os.path.join(voc_dir,'Annotations')
    for xml_file in tqdm(xml_files):
        if not xml_file.endswith('.xml'):
            continue

        start_time = time.time()
        tree = ET.parse(os.path.join(annotation_xml, xml_file))
        root = tree.getroot()

        # TODO 获取图片信息
        filename = root.find('filename').text
        image_id = os.path.splitext(filename)[0]
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        coco_output["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })

        # TODO 获取标注信息
        for obj in root.findall('object'):
            #TODO 得到点前的
            category_name = obj.find('name').text
            if category_name not in category_dict:
                continue
            #TODO 注意这里的类别索引是从1开始的
            category_id = category_dict[category_name]
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0,
            })

            annotation_id += 1

        # print(f'{xml_file} is done ... & time is: {time.time() - start_time}')

    with open(output_json_file, 'w') as out_file:
        json.dump(coco_output, out_file)


if __name__ == '__main__':
    # TODO PASCAL VOC标注文件路径
    voc_annotations_dir = r'D:\conda3\Transfer_Learning\PASCAL_VOC\VOCdevkit\VOC2007'
    output_coco_json_file = 'voc_to_coco_format.json'
    pascal_voc_to_coco(voc_annotations_dir, output_coco_json_file)
    print(f'Converted VOC annotations to COCO format and saved to {output_coco_json_file}')
    pass