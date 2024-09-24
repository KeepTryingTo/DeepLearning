"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/8/16-7:29
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import cv2
import time
import cvzone
import config
import numpy as np
from PIL import Image

import onnx
import onnxruntime

import torch
from torch import nn
from torchvision.ops import nms
from torchvision import models
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])

def checkONNX(onnxPath='onnx/SSDLite320_MobileNet_V3_Large.onnx'):
    #加载ONNX模型
    modelONNX = onnx.load(onnxPath)
    #检查模型格式是否正确
    onnx.checker.check_model(modelONNX)
    #打印ONNX的计算图
    # print(onnx.helper.printable_graph(modelONNX.graph))
    return modelONNX

def loadONNXObject():
    #TODO 输入大小为320
    #model = models.detection.ssdlite320_mobilenet_v3_large(
    #     weights=models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    #     , progress=True,num_classes=91
    # ).to(config.device)
    #TODO 输入大小为300
    #model = models.detection.ssd300_vgg16(
    #     weights=models.detection.SSD300_VGG16_Weights.DEFAULT,progress=True
    # )
    #TODO
    model = models.detection.fcos_resnet50_fpn(
        weights=models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT, progress=True
    )
    # TODO 注意这里的模型保存方式
    model.eval().to(config.device)
    x = torch.rand(size = (1,3,config.crop_size,config.crop_size)).to(config.device)
    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            'onnx/FCOS_ResNet50_FPN.onnx',  # 导出的ONNX名称
            do_constant_folding=True,#do_constant_folding=True 使静态图优化更彻底。
            opset_version=11,  # ONNX算子集版本
            input_names=['input'],  # 输入Tensor的名称（名称自己决定）
            output_names=['boxes','scores','labels']  # 输出tensor的名称（名称自己决定）
        )
    checkONNX(onnxPath='onnx/FCOS_ResNet50_FPN.onnx')

def out2org(box,crop_size,org_size):
    """
    :param box: [左上角和右下角]
    :param crop_size: 输入到模型中缩放的大小
    :param org_size: 原图大小[height,width]
    :return: 从缩放大小映射回相对原图大小的坐标框
    """
    left,top,right,bottom = box
    #TODO 首先将坐标映射到[0-1]之间
    left,top,right,bottom = left / crop_size,top / crop_size,right / crop_size, bottom / crop_size
    #TODO 最后映射到相对原图大小
    left,top,right,bottom = left * org_size[1],top * org_size[0],right * org_size[1],bottom * org_size[0]
    return [int(left),int(top),int(right),int(bottom)]

def drawRectangle(boxes, labels, scores, img_path,threshold):
    """
    :param boxes: 对应目标的坐标
    :param labels: 对应目标的标签
    :param scores: 对应目标的类别分数
    :return:
    """
    imgRe = cv2.imread(img_path)
    height,width,_ = imgRe.shape
    imgName = os.path.basename(img_path)
    for k in range(len(labels)):
        # 左上角坐标(xleft,yleft)和右下角坐标(xright,yright)
        box = out2org(boxes[k],config.crop_size,org_size=(height,width))
        xleft,yleft,xright,yright = box[0],box[1],box[2],box[3]

        class_id = labels[k].item()
        confidence = scores[k].item()
        # 这里只输出检测是人并且概率值最大的
        if confidence > threshold:
            text = config.className[class_id] + ': ' + str('{:.2f}%'.format(confidence * 100))
            cv2.rectangle(imgRe, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
            cvzone.putTextRect(img=imgRe, text=text, pos=(xleft + 9, yleft - 12),
                               scale=1, thickness=1, colorR=(0, 255, 0))
    cv2.imwrite(os.path.join(config.output,str(imgName)),imgRe)

    # cv2.imshow('img', imgRe)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def InferenceSignalImage(onnxPath = 'onnx/FCOS_ResNet50_FPN.onnx',img_path = 'images/horse01.png'):
    onnxSSDLite = onnxruntime.InferenceSession(onnxPath)
    # imgTo = cv2.imread(img_path)
    # imgTo = cv2.resize(imgTo,dsize=(config.crop_size,config.crop_size)) / 255
    # # 这里需要变换通道(H,W,C)=>(C,H,W)
    # # 方式一：
    # newImg = np.transpose(imgTo, axes = (2, 0, 1))
    # # 转换为tensor类型，这里如果使用torch.tensor(newImg)转换图像类型的话，
    # # 后面在输入网络时就会出错：RuntimeError: expected scalar type Double but found Float
    # newImg = torch.Tensor(newImg)
    # # 扩充维度，这里一定要注意，将其转化为numpy格式
    # newImg = torch.unsqueeze(input=newImg, dim=0).numpy()

    im0 = Image.open(img_path)
    width, height = im0.size

    im = im0.resize(size=(config.crop_size, config.crop_size))
    im = transform(im).unsqueeze(dim=0).to('cpu')
    newImg = im.numpy()

    input = {'input':newImg}
    # print("max value: {}".format(np.max(input)))
    # print("min value: {}".format(np.min(input)))
    #可能会报一个错误,关于这错误的一些讨论：https://github.com/microsoft/onnxruntime/issues/12669
    boxes,scores,labels = onnxSSDLite.run(output_names = ['boxes','scores','labels'],input_feed = input)

    boxes, labels, scores = torch.tensor(boxes), torch.tensor(labels), torch.tensor(scores)
    index = nms(boxes=boxes, scores=scores, iou_threshold=0.05)
    boxes = boxes[index]
    scores = scores[index]
    labels = labels[index]

    print('boxes.shape: {}'.format(np.shape(boxes)))
    print('labels.shape: {}'.format(np.shape(labels)))
    print('scores.shape: {}'.format(np.shape(scores)))

    # print('boxes: {}'.format(torch.tensor(boxes)))
    # print('labels: {}'.format(torch.tensor(labels)))
    # print('scores: {}'.format(torch.tensor(scores)))

    drawRectangle(boxes=boxes,labels=labels,scores=scores,img_path = img_path,threshold=0.5)

def InferenceTimeDetect(onnxPath = 'onnx/FCOS_ResNet50_FPN.onnx'):
    onnxSSDLite = onnxruntime.InferenceSession(onnxPath)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (config.crop_size, config.crop_size))
        newImg = frame / 255

        frame_ = cv2.flip(src=frame, flipCode=2)
        size = frame.shape
        # 这里需要变换通道(H,W,C)=>(C,H,W)
        # 方式一：
        newImg = np.transpose(newImg, (2, 0, 1))
        # 转换为tensor类型，这里如果使用torch.tensor(newImg)转换图像类型的话，
        # 后面在输入网络时就会出错：RuntimeError: expected scalar type Double but found Float
        newImg = torch.Tensor(newImg)
        newImg = torch.unsqueeze(input=newImg, dim=0).numpy()
        # 计算开始时间
        start_time = time.time()

        boxes,scores,labels = onnxSSDLite.run(output_names = ['boxes','scores','labels'],input_feed={'input':newImg})

        boxes,labels,scores = torch.tensor(boxes),torch.tensor(labels),torch.tensor(scores)
        index = nms(boxes=boxes, scores=scores, iou_threshold=0.1)
        boxes = boxes[index]
        scores = scores[index]
        labels = labels[index]

        for k in range(len(labels)):
            # 左上角坐标(xleft,yleft)和右下角坐标(xright,yright)
            xleft = int(boxes[k][0])
            yleft = int(boxes[k][1])
            xright = int(boxes[k][2])
            yright = int(boxes[k][3])

            class_id = labels[k].item()

            confidence = scores[k].item()
            # 这里只输出检测是人并且概率值最大的
            if confidence > 0.8:
                text = config.className[class_id] + ': ' + str('{:.2f}%'.format(confidence * 100))
                cv2.rectangle(frame, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
                cvzone.putTextRect(img=frame, text=text, pos=(xleft + 9, yleft - 12),
                                   scale=1, thickness=1, colorR=(0, 255, 0))


        # 计算结束时间
        end_time = time.time()
        FPS = round(1 / (end_time - start_time), 0)
        cv2.putText(img=frame, text='FPS: ' + str(FPS), org=(10, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(0, 255, 0), thickness=2)
        cv2.imshow('img', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # loadONNXObject()
    # InferenceSignalImage()
    images_list = os.listdir(config.root)
    for imgName in images_list:
        startTime = time.time()
        img_path = os.path.join(config.root,imgName)
        InferenceSignalImage(img_path=img_path)
        endTime = time.time()
        print('detect {} time is {}s'.format(imgName,endTime - startTime))

    # InferenceTimeDetect()
    pass