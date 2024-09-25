
import os
import cv2
import time
import cvzone
import config
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision.ops import nms
from torchvision import models
from torchvision import transforms

from torch.utils.mobile_optimizer import optimize_for_mobile

transform = transforms.Compose([
    transforms.Resize(size=(config.crop_size,config.crop_size)),
    transforms.ToTensor()
])

def pth2pt_f():
    # model = loadModel()
    model = models.detection.ssdlite320_mobilenet_v3_large(weights=models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
                                                           progress=True)
    model.eval()
    x = torch.zeros(size=(1,3,320,320))
    #https://blog.csdn.net/q7w8e9r4/article/details/135263015
    # jit_model = torch.jit.trace(model,x)
    jit_model = torch.jit.script(model)
    optimize_jit_model = optimize_for_mobile(jit_model)

    optimize_jit_model.save("./onnx/ssdlite320_mobilenet_v3_large.pt")

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


def detectImage(threshold = 0.5):
    model = torch.jit.load("./onnx/ssdlite320_mobilenet_v3_large.pt")
    images_list = os.listdir(config.root)

    for imgName in images_list:
        startTime = time.time()
        img_path = os.path.join(config.root,imgName)
        image = Image.open(img_path)
        #TODO PIL convert OpenCV(便于后面将框绘制到图像上)
        cv_img = np.array(image)
        cv_img = cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        height,width,_ = cv_img.shape

        image = transform(image).unsqueeze(dim=0).to(config.device)
        outs = model([image])
        #TODO NMS丢弃那些重叠的框，如果一个置信度最大的框和其他框之间的IoU > iou_threshold，那么就表示重叠并且需要丢弃
        indexs = nms(boxes=outs[0]['boxes'],scores=outs[0]['scores'],iou_threshold=0.05)
        endTime = time.time()
        print('boxes.shape: {}'.format(outs[0]['boxes'][indexs].shape))
        print('scores.shape: {}'.format(outs[0]['scores'][indexs].shape))
        print('labels.shape: {}'.format(outs[0]['labels'][indexs].shape))
        print('detect finished {} time is: {}s'.format(imgName,endTime - startTime))

        boxes = outs[0]['boxes'][indexs]
        scores = outs[0]['scores'][indexs]
        labels = outs[0]['labels'][indexs]

        for i in range(boxes.size()[0]):
            box = boxes[i]
            confidence = scores[i]
            label = labels[i]
            if confidence > threshold:
                box = [int(box[0]),int(box[1]),int(box[2]),int(box[3])]

                cv2.rectangle(
                    img=cv_img,pt1=(box[0],box[1]),pt2=(box[2],box[3]),
                    color=(0,255,0),thickness=1
                )

                text = "{} {}%".format(config.className[int(label.item())],round(confidence.item() * 100,2))
                cvzone.putTextRect(
                    img=cv_img,text=text,pos=(box[0],box[1] - 20),scale=1,thickness=1,colorR=(0,255,0),
                    font=cv2.FONT_HERSHEY_SIMPLEX
                )

        cv2.imwrite(os.path.join(config.output,imgName),cv_img)
        cv2.imshow('img',cv_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def timeDetect(threshold):
    model = torch.jit.load("./onnx/ssdlite320_mobilenet_v3_large.pt")
    # 计算开始时间
    start_time = time.time()
    # 计算帧率
    countFPS = 0
    # 开启摄像头
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(src=frame, dsize=(config.crop_size, config.crop_size))
        frame = cv2.flip(src=frame, flipCode=2)
        # 将其opencv读取的图像格式转换为PIL读取的类型格式
        frame_PIL = Image.fromarray(frame)
        img_transform = transform(frame_PIL)
        # 对图像进行升维
        img_Transform = torch.unsqueeze(input=img_transform, dim=0).to(config.device)

        detection = model([img_Transform])
        index = nms(boxes = detection[0]['boxes'],scores=detection[0]['scores'],iou_threshold=0.1)
        boxes = detection[0]['boxes'][index]
        labels = detection[0]['labels'][index]
        scores = detection[0]['scores'][index]

        # 获取类别概率值
        end_time = time.time()
        countFPS += 1
        FPS = round(countFPS / (end_time - start_time), 0)
        cv2.putText(img=frame, text='FPS: ' + str(FPS), org=(10, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(0, 255, 0), thickness=2)

        for k in range(len(labels)):
            xleft = int(boxes[k][0])
            yleft = int(boxes[k][1])
            xright = int(boxes[k][2])
            yright = int(boxes[k][3])

            class_id = labels[k].item()
            confidence = scores[k].item()

            if confidence > 0.3:
                text = config.className[class_id] + ': ' + str('{:.2f}%'.format(round(confidence * 100,2)))
                cv2.rectangle(frame, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
                cvzone.putTextRect(img=frame, text=text, pos=(xleft + 9, yleft - 12),
                                   scale=1, thickness=1, colorR=(0, 255, 0))
        cv2.imshow('img', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # pth2pt_f()
    detectImage()
    pass