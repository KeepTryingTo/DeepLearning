"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/13 15:46
"""

import os
import time
import cv2
# import cvzone
import numpy as np
import torch
from PIL import Image
from torch import nn
from configs import config
from configs.config import *
from utiles.nms import *
from torchvision import models
from models.modules import ConvBNRE,ConvBlock
from models.object.mydarknet import DarkNet
from models.object.mySelfModel import EDANet
from models.object.resnet50 import YOLOv1ResNet
from models.object.vgg_yolo import vgg16_bn
from models.object.darknet import Yolov1

# class_name = PERSON_CLASS
class_name =VOC_CLASSES
weight_path = r'weights/5.538_resnet_losses_t_best_model.pth.tar'

#TODO load model
def loadModel():
    ##########################################################################################
    # model = DarkNet(
    #     in_channels=3,img_size=IMG_SIZE,channels_list=CHANNELS_LIST,
    #     num_classes=VOC_NUM_CLASSES,S = S,B = B
    # )
    ##########################################################################################
    model = YOLOv1ResNet(
        B = B,S = S,C = VOC_NUM_CLASSES
    )
    ##########################################################################################
    # model = EDANet(num_classes=VOC_NUM_CLASSES,B = B,S = S)
    ##########################################################################################
    # model = Yolov1(in_channels=3,split_size = S,
    #               num_boxes = B,num_classes = VOC_NUM_CLASSES)
    ##########################################################################################
    # model = vgg16_bn()
    # vgg = models.vgg16_bn(pretrained=True)
    # new_state_dict = vgg.state_dict()
    # dd = model.state_dict()
    # for k in new_state_dict.keys():
    #     # print(k)
    #     if k in dd.keys() and k.startswith('features'):
    #         # print('yes')
    #         dd[k] = new_state_dict[k]
    # model.load_state_dict(dd)
    ##########################################################################################
    checkpoint = torch.load(weight_path,map_location='cpu')
    print(checkpoint['model'].keys())
    model.load_state_dict(checkpoint['model'])
    print('load model is done ...')
    return model

def predictSingleImage(conf_threshold = 0.25):
    model = loadModel()
    model.eval()
    root = r'images'
    transform = config.transform
    for img_name in os.listdir(root):
        img_path = os.path.join(root,img_name)
        img = cv2.imread(img_path)
        H,W,C = img.shape
        imgTo = cv2.resize(img,dsize=(IMG_SIZE,IMG_SIZE))
        imgTo = cv2.cvtColor(imgTo,cv2.COLOR_BGR2RGB)
        imgTo = Image.fromarray(imgTo)
        t_img = transform(imgTo).unsqueeze(0)

        start_time = time.time()
        outputs = model(t_img)
        #得到经过初步筛选的框
        boxes,probs,cls_indexes,is_exist_object = convert_cellboxes(
            predictions=outputs,S = S,B = B,num_classes=VOC_NUM_CLASSES,
            conf_threshold=CONF_THRESHOLD,iou_threshold=IOU_THRESHOLD
        )
        #进行non_max_suppression算法多虑掉那些多余或者重叠的框
        boxes,scores,class_label = nms(
            boxes=boxes,probs=probs,cls_indexes=cls_indexes,
            iou_threshold=IOU_THRESHOLD
        )
        print('number boxes: {}'.format(boxes.size()[0]))
        print('scores: {}'.format(scores))
        print('boxes: {}'.format(boxes))
        print('cls_label: {}'.format(class_label))
        end_time = time.time()
        if is_exist_object is True:
            for i,(score,box) in enumerate(zip(scores,boxes)):
                if score > conf_threshold:
                    cls_name = class_name[class_label[i].item()]
                    #得到经过NMS之后的框[cx,cy,w,h]
                    xmin,ymin,xmax,ymax = box
                    xmin,xmax,ymin,ymax = xmin.clamp(0,W),xmax.clamp(0,W),ymin.clamp(0,H),ymax.clamp(0,H)
                    #将其坐标还原回原图的大小
                    xmin,ymin = int(xmin * W),int(ymin * H)
                    xmax,ymax = int(xmax * W),int(ymax * H)
                    xmin = np.minimum(xmin, W)
                    ymin = np.minimum(ymin, H)
                    xmax = np.minimum(xmax, W)
                    ymax = np.minimum(ymax, H)
                    #绘制框和类别信息
                    # print(xmin,ymin,xmax,ymax)
                    cv2.rectangle(img=img,pt1=(xmin,ymin),pt2=(xmax,ymax),color=(0,255,255),thickness=1)
                    text = str(cls_name) + ":" + str(round(score.item() * 100,1)) + "%"
                    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    # cvzone.putTextRect(
                    #     img,text=text,pos=(int(xmin),int(ymin) + baseline),
                    #     scale=1,thickness=1,colorT=(0,255,0)
                    # )
        print('inference time : {}'.format(end_time-start_time))
        cv2.imwrite(r'runs/{}'.format(img_name),img)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)

    cv2.destroyAllWindows()

def realTimeDetect(conf_threshold = 0.25):
    model = loadModel()
    model.eval()
    root = r'images'
    transform = config.transform
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret,frame = cap.read()
        H,W,C = frame.shape
        imgTo = cv2.resize(frame,dsize=(IMG_SIZE,IMG_SIZE))
        imgTo = cv2.cvtColor(imgTo,cv2.COLOR_BGR2RGB)
        imgTo = Image.fromarray(imgTo)
        t_img = transform(imgTo).unsqueeze(0)

        start_time = time.time()
        outputs = model(t_img)
        #得到经过初步筛选的框
        boxes,probs,cls_indexes,is_exist_object = convert_cellboxes(
            predictions=outputs,S = S,B = 2,num_classes=VOC_NUM_CLASSES,
            conf_threshold=CONF_THRESHOLD,iou_threshold=IOU_THRESHOLD
        )
        #进行non_max_suppression算法多虑掉那些多余或者重叠的框
        boxes,scores,class_label = nms(
            boxes=boxes,probs=probs,cls_indexes=cls_indexes,
            iou_threshold=IOU_THRESHOLD
        )
        print('number boxes: {}'.format(boxes.size()[0]))
        print('scores: {}'.format(scores))
        print('boxes: {}'.format(boxes))
        end_time = time.time()
        if is_exist_object:
            for i,(score,box) in enumerate(zip(scores,boxes)):
                if score > conf_threshold:
                    cls_name = class_name[class_label[i].item()]
                    # 得到经过NMS之后的框[cx,cy,w,h]
                    xmin, ymin, xmax, ymax = box
                    xmin, xmax, ymin, ymax = xmin.clamp(0, W), xmax.clamp(0, W), ymin.clamp(0, H), ymax.clamp(0, H)
                    # 将其坐标还原回原图的大小
                    xmin, ymin = int(xmin * W), int(ymin * H)
                    xmax, ymax = int(xmax * W), int(ymax * H)
                    xmin = np.minimum(xmin, W)
                    ymin = np.minimum(ymin, H)
                    xmax = np.minimum(xmax, W)
                    ymax = np.minimum(ymax, H)
                    #绘制框和类别信息
                    # print(xmin,ymin,xmax,ymax)
                    cv2.rectangle(img=frame,pt1=(xmin,ymin),pt2=(xmax,ymax),color=(0,255,255),thickness=1)
                    text = str(cls_name) + ":" + str(round(score.item() * 100,1)) + "%"
                    # cvzone.putTextRect(
                    #     frame,text=text,pos=(int(xmin),int(ymin) - 10),
                    #     scale=1,thickness=1,colorT=(0,255,0)
                    # )
        print('inference time : {}'.format(end_time-start_time))
        cv2.imshow('img',frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    predictSingleImage()
    # realTimeDetect()
    pass
