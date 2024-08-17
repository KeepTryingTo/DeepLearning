"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/8/17-16:38
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import os
import cv2
import cvzone
import numpy as np
import onnxruntime
from PIL import Image

import torch
from torch import nn
from torchvision import transforms

from utils.general import non_max_suppression


from utils.augmentations import letterbox
from models.common import DetectMultiBackend

weights = r'yolov5s.pt'
device = 'cpu' if torch.cuda.is_available() else 'cpu'
data = r'data/coco128.yaml'
root = r'images'
img_size = 640

conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=100  # maximum detections per image

classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS

model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt

transform = transforms.Compose([
    transforms.ToTensor()
])

def detectImage01():
    images_list = os.listdir(root)
    for imgName in images_list:
        img_path = os.path.join(root,imgName)
        im0 = cv2.imread(img_path)
        im0 = letterbox(im0,img_size,stride = stride,auto = pt)[0]

        height,width,_ = im0.shape
        print('height = {}, width = {}'.format(height,width))

        im = im0.transpose((2,0,1))[::-1]
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(model.device).unsqueeze(dim = 0)
        im = im.half() if model.fp16 else im.float()
        im /= 255

        pred = model(im,augment=False, visualize=False)

        print('pred.shape: {}'.format(pred[0].shape))
        pred = non_max_suppression(prediction=pred, conf_thres=conf_thres, iou_thres=iou_thres,
                                   classes=classes, agnostic=agnostic_nms, max_det=max_det)
        print('NMS pred.shape: {}'.format(pred[0].shape))
        frame = np.array(im0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        predictions = pred[0]

        for k in range(predictions.size()[0]):
            # 左上角坐标(xleft,yleft)和右下角坐标(xright,yright)
            xleft = int(predictions[k][0])
            yleft = int(predictions[k][1])
            xright = int(predictions[k][2])
            yright = int(predictions[k][3])

            confidence = predictions[k][4].item()
            class_id = predictions[k][5].item()

            # 这里只输出检测是人并且概率值最大的
            if confidence > conf_thres:
                text = names[class_id] + ': ' + str('{:.2f}%'.format(confidence * 100))
                cv2.rectangle(frame, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
                cvzone.putTextRect(img=frame, text=text, pos=(xleft + 9, yleft - 12),
                                   scale=1, thickness=1, colorR=(0, 255, 0))
        # cv2.imwrite(os.path.join('outputs', imgName), frame)
        # cv2.imshow('img',frame)
        # cv2.waitKey(0)
        # break

def detectImage02():
    images_list = os.listdir(root)
    for imgName in images_list:
        img_path = os.path.join(root, imgName)
        im0 = Image.open(img_path)
        width,height = im0.size

        im = im0.resize(size=(img_size,img_size))
        im = transform(im).unsqueeze(dim = 0).to(device)
        im = im.half() if model.fp16 else im.float()

        pred = model(im, augment=False, visualize=False)
        print('pred.shape: {}'.format(pred[0].shape))
        pred = non_max_suppression(prediction=pred, conf_thres=conf_thres, iou_thres=iou_thres,
                                   classes=classes, agnostic=agnostic_nms, max_det=max_det)
        print('NMS pred.shape: {}'.format(pred[0].shape))

        #TODO PIL convert Opencv
        frame = np.array(im0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        predictions = pred[0]
        for k in range(predictions.size()[0]):
            # 左上角坐标(xleft,yleft)和右下角坐标(xright,yright)
            xleft = int(predictions[k][0] / img_size * width)
            yleft = int(predictions[k][1] / img_size * height)
            xright = int(predictions[k][2] / img_size * width)
            yright = int(predictions[k][3] / img_size * height)

            confidence = predictions[k][4].item()
            class_id = predictions[k][5].item()

            # 这里只输出检测是人并且概率值最大的
            if confidence > conf_thres:
                text = names[class_id] + ': ' + str('{:.2f}%'.format(confidence * 100))
                cv2.rectangle(frame, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
                cvzone.putTextRect(img=frame, text=text, pos=(xleft + 9, yleft - 12),
                                   scale=1, thickness=1, colorR=(0, 255, 0))
        cv2.imwrite(os.path.join('outputs', imgName), frame)
        cv2.imshow('img', frame)
        cv2.waitKey(0)

def timeDetect():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret,frame = cap.read()

        im0 = cv2.resize(frame,dsize=(img_size, img_size))
        im = Image.fromarray(im0)
        im = transform(im).unsqueeze(dim=0).to(device)
        im = im.half() if model.fp16 else im.float()

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(prediction=pred, conf_thres=conf_thres, iou_thres=iou_thres,
                                   classes=classes, agnostic=agnostic_nms, max_det=max_det)

        predictions = pred[0]
        for k in range(predictions.size()[0]):
            # 左上角坐标(xleft,yleft)和右下角坐标(xright,yright)
            xleft = int(predictions[k][0])
            yleft = int(predictions[k][1])
            xright = int(predictions[k][2])
            yright = int(predictions[k][3])

            confidence = predictions[k][4].item()
            class_id = predictions[k][5].item()

            # 这里只输出检测是人并且概率值最大的
            if confidence > conf_thres:
                text = names[class_id] + ': ' + str('{:.2f}%'.format(confidence * 100))
                cv2.rectangle(frame, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
                cvzone.putTextRect(img=frame, text=text, pos=(xleft + 9, yleft - 12),
                                   scale=1, thickness=1, colorR=(0, 255, 0))
        cv2.imshow('img', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def onnxDetectImage():
    images_list = os.listdir(root)
    onnxYolov5 = onnxruntime.InferenceSession("yolov5s.onnx")
    for imgName in images_list:
        img_path = os.path.join(root, imgName)
        im0 = Image.open(img_path)
        width, height = im0.size

        im = im0.resize(size=(img_size, img_size))
        im = transform(im).unsqueeze(dim=0).to(device)
        im = im.half() if model.fp16 else im.float()
        im = im.numpy()

        pred = onnxYolov5.run(output_names=['output0'],input_feed={'images':im})
        print('pred.shape: {}'.format(pred[0].shape))
        pred = torch.tensor(pred)
        pred = non_max_suppression(prediction=pred, conf_thres=conf_thres, iou_thres=iou_thres,
                                   classes=classes, agnostic=agnostic_nms, max_det=max_det)
        print('NMS pred.shape: {}'.format(pred[0].shape))

        # TODO PIL convert Opencv
        frame = np.array(im0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        predictions = pred[0]
        for k in range(predictions.size()[0]):
            # 左上角坐标(xleft,yleft)和右下角坐标(xright,yright)
            xleft = int(predictions[k][0] / img_size * width)
            yleft = int(predictions[k][1] / img_size * height)
            xright = int(predictions[k][2] / img_size * width)
            yright = int(predictions[k][3] / img_size * height)

            confidence = predictions[k][4].item()
            class_id = predictions[k][5].item()

            # 这里只输出检测是人并且概率值最大的
            if confidence > conf_thres:
                text = names[class_id] + ': ' + str('{:.2f}%'.format(confidence * 100))
                cv2.rectangle(frame, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
                cvzone.putTextRect(img=frame, text=text, pos=(xleft + 9, yleft - 12),
                                   scale=1, thickness=1, colorR=(0, 255, 0))
        cv2.imwrite(os.path.join('outputs', imgName), frame)
        # cv2.imshow('img', frame)
        # cv2.waitKey(0)


if __name__ == '__main__':
    # detectImage01()
    # detectImage02()
    # timeDetect()
    onnxDetectImage()
    pass