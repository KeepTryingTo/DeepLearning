"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/9/25-16:33
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import cv2
import cvzone
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.mobile_optimizer import optimize_for_mobile

from torchvision import transforms
from torchvision.ops import nms
from models.common import DetectMultiBackend

transform = transforms.Compose([
    transforms.ToTensor()
])

weights = r'yolov5s.pt'
device = 'cpu' if torch.cuda.is_available() else 'cpu'
data = r'data/coco128.yaml'
root = r'images'
img_size = 640

conf_thres=0.25  # confidence threshold
iou_thres=0.20  # NMS IOU threshold
max_det=100  # maximum detections per image

model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt

def convertPT():
    x = torch.rand(size=(1,3,img_size,img_size))
    jit_model = torch.jit.script(model,x)
    optimize_jit_model = optimize_for_mobile(jit_model)
    optimize_jit_model.save("yolov5s.torchscript")
    print("convert is finished!")

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
def onnxDetectImage():
    images_list = os.listdir(root)
    model = torch.jit.load("yolov5s.torchscript")
    for imgName in images_list:
        img_path = os.path.join(root, imgName)
        im0 = Image.open(img_path)
        width, height = im0.size

        im = im0.resize(size=(img_size, img_size))
        im = transform(im).unsqueeze(dim=0).to(device)

        predictions = model(im)
        #[1,25200,85] = [1,25200,xywh + conf + 80]
        print('predictions.shape: {}'.format(predictions[0].shape))
        pred = np.squeeze(predictions)
        boxes = xywh2xyxy(pred[...,:4])
        confidences = pred[...,4]
        cls_prob = pred[...,5:85]
        labels = np.argmax(cls_prob,axis=-1)
        confidences = confidences * np.max(cls_prob,axis=-1)
        boxes , confidences, labels = torch.tensor(boxes,dtype=torch.float32),torch.tensor(confidences,dtype=torch.float32),torch.tensor(labels)
        indexs = nms(boxes = boxes,scores=confidences,iou_threshold=iou_thres)

        boxes = boxes[indexs]
        confidences = confidences[indexs]
        labels = labels[indexs]

        # TODO PIL convert Opencv
        frame = np.array(im0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for k in range(boxes.size()[0]):
            # 左上角坐标(xleft,yleft)和右下角坐标(xright,yright)
            xleft = int(boxes[k][0] / img_size * width)
            yleft = int(boxes[k][1] / img_size * height)
            xright = int(boxes[k][2] / img_size * width)
            yright = int(boxes[k][3] / img_size * height)

            confidence = confidences[k].item()
            class_id = labels[k].item()

            # 这里只输出检测是人并且概率值最大的
            if confidence > conf_thres:
                text = names[class_id] + ': ' + str('{:.2f}%'.format(confidence * 100))
                cv2.rectangle(frame, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
                cvzone.putTextRect(img=frame, text=text, pos=(xleft + 9, yleft - 12),
                                   scale=1, thickness=1, colorR=(0, 255, 0))
        cv2.imwrite(os.path.join('outputs', imgName), frame)
        # cv2.imshow('img', frame)
        # cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # detectImage01()
    # detectImage02()
    # timeDetect()
    convertPT()
    onnxDetectImage()
    pass