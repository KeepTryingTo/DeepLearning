"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/11/27-22:13
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

annotations = [
    {
        'filename': 'image1.jpg',
        'width': 1920,
        'height': 1080,
        'objects': [
            {'name': 'cat', 'bbox': [100, 150, 200, 300]},  # [x_min, y_min, x_max, y_max]
            {'name': 'dog', 'bbox': [300, 400, 500, 600]}
        ]
    },
    # 添加更多图像的标注信息
]

import xml.etree.ElementTree as ET
import os

def create_voc_xml(annotations, output_dir):
    for annotation in annotations:
        # 创建根元素
        annotation_elem = ET.Element('annotation')

        # 添加文件名、尺寸等信息
        ET.SubElement(annotation_elem, 'folder').text = 'images'
        ET.SubElement(annotation_elem, 'filename').text = annotation['filename']
        ET.SubElement(annotation_elem, 'path').text = os.path.join(output_dir, annotation['filename'])

        # 添加图像的宽度和高度
        size_elem = ET.SubElement(annotation_elem, 'size')
        ET.SubElement(size_elem, 'width').text = str(annotation['width'])
        ET.SubElement(size_elem, 'height').text = str(annotation['height'])
        ET.SubElement(size_elem, 'depth').text = '3'  # 通常是 RGB 彩色图像

        # 添加目标信息
        for obj in annotation['objects']:
            object_elem = ET.SubElement(annotation_elem, 'object')
            ET.SubElement(object_elem, 'name').text = obj['name']
            ET.SubElement(object_elem, 'pose').text = 'Unspecified'
            ET.SubElement(object_elem, 'truncated').text = '0'
            ET.SubElement(object_elem, 'difficult').text = '0'
            bndbox_elem = ET.SubElement(object_elem, 'bndbox')

            # 添加边界框坐标
            ET.SubElement(bndbox_elem, 'xmin').text = str(obj['bbox'][0])
            ET.SubElement(bndbox_elem, 'ymin').text = str(obj['bbox'][1])
            ET.SubElement(bndbox_elem, 'xmax').text = str(obj['bbox'][2])
            ET.SubElement(bndbox_elem, 'ymax').text = str(obj['bbox'][3])

        # 生成 XML 树并写入文件
        tree = ET.ElementTree(annotation_elem)
        output_file = os.path.join(output_dir, annotation['filename'].replace('.jpg', '.xml'))
        tree.write(output_file, encoding='utf-8', xml_declaration=True)

# 示例用法
output_directory = './annotations'  # 替换为你的输出目录
os.makedirs(output_directory, exist_ok=True)
create_voc_xml(annotations, output_directory)