"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/10/25-9:25
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import os
import time
import cv2
# import cvzone
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import transforms

import config as cfg
from retinanet import model
from prepare_data import Resizer,Normalizer,VOC_CLASSES

root = r'./images'
save = r'./outputs/'
weight_path = r'weights/voc_retinanet_90.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO 加载预训练模型用于验证
eval_model = torch.load(weight_path, map_location=torch.device('cpu'))
eval_model = torch.nn.DataParallel(eval_model)
eval_model.eval()
eval_model.to(device)

img_size = (608,608)

transform = transforms.Compose([
    transforms.Resize(size=img_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    )
])

name_2_label = dict(
    zip(VOC_CLASSES, range(len(VOC_CLASSES)))
)
label_2_name = {
    v: k for k, v in name_2_label.items()
}

def predictImage(conf_threshold = 0.55):
    img_list = os.listdir(root)

    for img_name in img_list:
        start_time = time.time()
        img = Image.open(os.path.join(root,img_name))
        w,h = img.size
        x = transform(img).unsqueeze(dim = 0)

        with torch.no_grad():
            scores, classification, transformed_anchors = eval_model(x)
        idxs = np.where(scores.cpu() > conf_threshold)

        boxes = transformed_anchors[idxs]
        scores = scores[idxs]
        labels = classification[idxs]
        print('boxes:', boxes.size())

        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for i in range(boxes.shape[0]):
            bbox = boxes[i]
            x1 = int(bbox[0] / img_size[0] * w)
            y1 = int(bbox[1] / img_size[1] * h)
            x2 = int(bbox[2] / img_size[0] * w)
            y2 = int(bbox[3] / img_size[1] * h)

            cv2.rectangle(img=cv_img, pt1=(x1, y1), pt2=(x2, y2),
                          color=(255, 0, 255), thickness=2, lineType=2)
            cv2.putText(cv_img, VOC_CLASSES[int(labels[i])]+"%.2f"%scores[i], (x1, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 255))

            text = "{} {}%".format(VOC_CLASSES[int(labels[i])], round(scores[i].item() * 100, 2))
            # cvzone.putTextRect(
            #     img=cv_img, text=text, pos=(x1 + 9, y1 - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
            #     font=cv2.FONT_HERSHEY_SIMPLEX
            # )
        end_time = time.time()
        print('{} inference time: {} seconds'.format(img_name,end_time - start_time))
        cv2.imwrite(os.path.join(save,img_name),cv_img)


def timeDetect():

    cap = cv2.VideoCapture(0)
    count = 0
    start_time = time.time()

    while cap.isOpened():
        ret,frame = cap.read()
        count += 1
        if ret == False:
            break

        frame = cv2.resize(frame,dsize=(800,600))
        frame = cv2.flip(src=frame, flipCode=2)
        h, w = np.shape(frame)[:2]
        scale = torch.Tensor([w, h, w, h])

        with torch.no_grad():
            img = Image.fromarray(frame)
            x = transform(img).unsqueeze(0)
            if device:
                x = x.to(device)
                scale = scale.to(device)
        with torch.no_grad():
            scores, classification, transformed_anchors = eval_model(x)
        idxs = np.where(scores.cpu() > 0.1)

        boxes = transformed_anchors[idxs]
        scores = scores[idxs]
        labels = classification[idxs]
        print('boxes:', boxes.size())

        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for i in range(boxes.shape[0]):
            bbox = boxes[i]
            x1 = int(bbox[0] / img_size[0] * w)
            y1 = int(bbox[1] / img_size[1] * h)
            x2 = int(bbox[2] / img_size[0] * w)
            y2 = int(bbox[3] / img_size[1] * h)

            cv2.rectangle(img=cv_img, pt1=(x1, y1), pt2=(x2, y2),
                          color=(255, 0, 255), thickness=2, lineType=2)
            cv2.putText(cv_img, VOC_CLASSES[int(labels[i])] + "%.2f" % scores[i], (x1, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 255))

            text = "{} {}%".format(VOC_CLASSES[int(labels[i])], round(scores[i].item() * 100, 2))
            # cvzone.putTextRect(
            #     img=cv_img, text=text, pos=(x1 + 9, y1 - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
            #     font=cv2.FONT_HERSHEY_SIMPLEX
            # )
        cv2.imshow('img',cv_img)
        key = cv2.waitKey(1)
        if key & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    predictImage()
    # timeDetect()
    pass