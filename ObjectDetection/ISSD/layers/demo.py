"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/11/7-21:33
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import numpy as np
import cv2

def IoU(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0
def NMW(boxes, confidences):
    n = len(boxes)
    weights = np.zeros(n)
    # TODO 找到置信度最高的框
    max_idx = np.argmax(confidences)
    max_box = boxes[max_idx]
    # TODO 计算每个框的权重
    for i in range(n):
        weights[i] = confidences[i] * IoU(boxes[i], max_box)
    # TODO 计算加权平均框
    weighted_box = np.sum(weights[:, np.newaxis] * boxes, axis=0) / np.sum(weights)

    return weighted_box

def main():
    # 示例候选框和置信度
    boxes = np.array([
        [50, 50, 150, 150],  # Box 1
        [60, 60, 140, 140],  # Box 2 (overlapping)
        [70, 70, 130, 130],  # Box 3 (more overlapping)
        [200, 200, 300, 300]  # Box 4 (not overlapping)
    ])
    confidences = np.array([0.9, 0.8, 0.85, 0.6])  # 置信度
    # TODO 使用 NMW 过滤边界框
    final_box = NMW(boxes, confidences)
    print("Final Weighted Box:", final_box)
    img = np.zeros((400, 400, 3), dtype=np.uint8)  # 创建一张空白图像
    # TODO 首先绘制所有框
    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    # TODO 绘制经过NMW计算之后的结果图
    cv2.rectangle(img, (int(final_box[0]), int(final_box[1])),
                  (int(final_box[2]), int(final_box[3])),
                  (0, 255, 0), 2)

    cv2.imshow('Image with Bounding Boxes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
