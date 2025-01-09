"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/10/18-19:37
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

from __future__ import print_function
import os
import cv2
import yaml
import cvzone
import argparse
import time
import numpy as np
from PIL import Image
import torch
from commons.augmentations import ScalePadding
from nets.centernet import CenterNet

rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

coco_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

device = 'cpu' if torch.cuda.is_available() else 'cpu'

with open("config/centernet.yaml", 'r') as rf:
    cfg = yaml.safe_load(rf)
model_cfg = cfg['model']
model = CenterNet(num_cls=model_cfg['num_cls'],
                  PIXEL_MEAN=model_cfg['PIXEL_MEAN'],
                  PIXEL_STD=model_cfg['PIXEL_STD'],
                  backbone=model_cfg['backbone'],
                  cfg=model_cfg)

weights = torch.load("weights/0_TTFNet_best_map.pth",map_location='cpu')['ema']
model.load_state_dict(weights)
model.to(device)
model.eval()
print('load model is done ...')

img_size = 512


COCO_CLASSES = (
    'person','bicycle',
    'car','motorcycle','airplane','bus',
    'train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter',
    'bench','bird','cat','dog','horse','sheep',
    'cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase',
    'frisbee','skis','snowboard','sports ball','kite',
    'baseball bat','baseball glove','skateboard',
    'surfboard','tennis racket','bottle','wine glass',
    'cup','fork','knife','spoon','bowl','banana',
    'apple','sandwich','orange','broccoli','carrot',
    'hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table',
    'toilet','tv','laptop','mouse','remote',
    'keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock',
    'vase','scissors','teddy bear','hair drier',
    'toothbrush'
)
def predictImage(
        save_folder,
        device,
        conf_threshod=0.65,
):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    root = r"./images"
    img_list = os.listdir(root)

    # TODO 对所有的图像进行检测
    for imgName in img_list:
        start_time = time.time()

        img_path = os.path.join(root,imgName)
        img = cv2.imread(img_path)
        h,w,_ = img.shape
        scale = torch.Tensor([w,h,w,h])
        img_out = cv2.resize(img, dsize=(img_size,img_size))
        img_out = img_out[:, :, [2, 1, 0]].astype(np.float32) / 255.0
        img_out = ((img_out - np.array(rgb_mean)) / np.array(rgb_std)).transpose(2, 0, 1).astype(np.float32)
        img_out = torch.from_numpy(np.ascontiguousarray(img_out)).unsqueeze(0).float().to(device)
        predicts = model(img_out)  # list(predict) predict.shape=[num_box,6]  6==>x1,y1,x2,y2,score,label

        for i in range(len(predicts)):
            predicts[i][:, [0, 2]] = predicts[i][:, [0, 2]].clamp(min=0, max=w)
            predicts[i][:, [1, 3]] = predicts[i][:, [1, 3]].clamp(min=0, max=h)

        box = predicts[0]
        if box is None:
            continue
        box[:, [0, 2]] = box[:, [0, 2]] / img_size * w
        box[:, [1, 3]] = box[:, [1, 3]] / img_size * h
        box = box.detach().cpu().numpy()
        boxes = box[:, :4]
        scores = box[:,4]
        lables = box[:,5]

        end_time = time.time()
        print('output.shape: {}'.format(np.shape(boxes)))

        #TODO 遍历所有的类别，除了背景之外
        for i in range(np.shape(boxes)[0]):
            #TODO 获得当前类的所有box和置信度
            box = boxes[i]
            confidence = scores[i]
            label = lables[i]
            #TODO 对于低置信度和背景都过滤掉
            if confidence > conf_threshod:
                box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                cv2.rectangle(
                    img=img, pt1=(box[0], box[1]), pt2=(box[2], box[3]),
                    color=(255, 0, 255), thickness=1
                )
                text = "{} {}%".format(COCO_CLASSES[int(label)], round(confidence.item() * 100, 2))
                cvzone.putTextRect(
                    img=img, text=text, pos=(box[0] + 9, box[1] - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
                    font=cv2.FONT_HERSHEY_SIMPLEX
                )

        cv2.imwrite(os.path.join(save_folder, imgName), img)
        print('inference time : {}'.format(time.time() - start_time))
        # cv2.imshow('img',cv_img)
        # cv2.waitKey(0)

    cv2.destroyAllWindows()

def timeDetect(
    device,
    conf_threshod=0.35,
):

    cap = cv2.VideoCapture(0)
    count = 0
    start_time = time.time()

    while cap.isOpened():
        frame, ret = cap.read()
        count += 1
        if ret == False:
            break

        frame = cv2.resize(frame, dsize=(800, 600))
        frame = cv2.flip(src=frame, flipCode=2)
        h, w = np.shape(frame)[:2]

        h, w, _ = frame.shape
        scale = torch.Tensor([w, h, w, h])
        h, w = frame.shape[:2]
        img_out = cv2.resize(frame, dsize=(img_size,img_size))
        img_out = img_out[:, :, [2, 1, 0]].astype(np.float32) / 255.0
        img_out = ((img_out - np.array(rgb_mean)) / np.array(rgb_std)).transpose(2, 0, 1).astype(np.float32)
        img_out = torch.from_numpy(np.ascontiguousarray(img_out)).unsqueeze(0).float().to(device)
        predicts = model(img_out)  # list(predict) predict.shape=[num_box,6]  6==>x1,y1,x2,y2,score,label

        for i in range(len(predicts)):
            predicts[i][:, [0, 2]] = predicts[i][:, [0, 2]].clamp(min=0, max=w)
            predicts[i][:, [1, 3]] = predicts[i][:, [1, 3]].clamp(min=0, max=h)

        box = predicts[0]
        if box is None:
            continue
        box[:, [0, 2]] = box[:, [0, 2]] / img_size * w
        box[:, [1, 3]] = box[:, [1, 3]] / img_size * h
        box = box.detach().cpu().numpy()

        boxes = box[:, :4]
        scores = box[:, 4]
        lables = box[:, 5]

        end_time = time.time()
        print('output.shape: {}'.format(np.shape(boxes)))

        # TODO 遍历所有的类别，除了背景之外
        for i in range(np.shape(boxes)[0]):
            # TODO 获得当前类的所有box和置信度
            box = boxes[i]
            confidence = scores[i]
            label = lables[i]
            # TODO 对于低置信度和背景都过滤掉
            if confidence > conf_threshod:
                box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                cv2.rectangle(
                    img=frame, pt1=(box[0], box[1]), pt2=(box[2], box[3]),
                    color=(255, 0, 255), thickness=1
                )
                text = "{} {}%".format(COCO_CLASSES[int(i)], round(confidence.item() * 100, 2))
                cvzone.putTextRect(
                    img=frame, text=text, pos=(box[0] + 9, box[1] - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
                    font=cv2.FONT_HERSHEY_SIMPLEX
                )

        cv2.imshow('img',frame)
        key = cv2.waitKey(1)
        if key & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    save_folder = r'./outputs/'
    predictImage(save_folder=save_folder,
                 device=device,
                 conf_threshod=0.35)

# https://github.com/KeepTryingTo/DeepLearning.git
