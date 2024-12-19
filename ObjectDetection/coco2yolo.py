"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/19-10:25
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import json
import cv2
import numpy as np


def convert_coco_to_yolo(coco_json_path, images_dir, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        # 读取 COCO JSON 文件
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

        # 获取类别信息
    categories = {category['id']: category['name'] for category in coco_data['categories']}
    category_mapping = {name: idx for idx, name in enumerate(categories.values())}

    # 遍历每个图像
    for image in coco_data['images']:
        image_id = image['id']
        image_file_name = image['file_name']

        # 获取对应的图像尺寸
        width = image['width']
        height = image['height']

        # todo 创建 YOLO 格式的标签文件
        yolo_labels = []

        # todo 查找当前图像的注释
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                # todo 获取边界框信息
                bbox = annotation['bbox']  # [x_min, y_min, width, height]
                class_id = category_mapping[categories[annotation['category_id']]]

                # todo 计算 YOLO 格式
                x_center = bbox[0] + bbox[2] / 2
                y_center = bbox[1] + bbox[3] / 2
                x_center /= width
                y_center /= height
                bbox_width = bbox[2] / width
                bbox_height = bbox[3] / height

                # todo 添加到 YOLO 标签列表
                yolo_labels.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")

        # TODO 保存 YOLO 标签到文本文件
        yolo_label_file = os.path.join(output_dir, image_file_name.replace('.jpg', '.txt'))
        with open(yolo_label_file, 'w') as label_file:
            label_file.write('\n'.join(yolo_labels))

            # 复制图像到输出目录（可选，视需要而定）
        src_image_path = os.path.join(images_dir, image_file_name)
        dst_image_path = os.path.join(output_dir, image_file_name)
        if os.path.exists(src_image_path):
            cv2.imwrite(dst_image_path, cv2.imread(src_image_path))

        print('{} is finished!'.format(image_id))

if __name__ == "__main__":
    # 示例用法
    coco_json_path = '/home/ff/myProject/KGT/myProjects/myDataset/coco/captions/annotations/instances_train2014.json'  # COCO 的 g注释 JSON 文件路径
    images_dir = '/home/ff/myProject/KGT/myProjects/myDataset/coco/train2014/train2014'  # 包含图像的目录
    output_dir = '/home/ff/myProject/KGT/myProjects/myDataset/coco/yolo'  # 输出 YOLO 标签和图像的目录

    convert_coco_to_yolo(coco_json_path, images_dir, output_dir)