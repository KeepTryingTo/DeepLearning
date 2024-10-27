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
import cvzone
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import transforms

import config as cfg
from assd_resnet101 import ASSD_ResNet101

root = r'./images'
save = r'D:\conda3\Transfer_Learning\ObjectDetect\ASSD-Pytorch-master\runs'
weight_path = r'weights/75_0.0568_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO 加载预训练模型用于验证
eval_model = ASSD_ResNet101(num_classes=cfg.num_classes,
                            num_blocks=cfg.mbox,
                            top_k=cfg.top_k,
                            conf_thresh=cfg.conf_thresh,
                            nms_thresh=cfg.nms_thresh,
                            variance=cfg.variance)
checkpoint = torch.load(weight_path, map_location='cpu')
eval_model.load_state_dict(checkpoint)
eval_model = eval_model.to(device)
eval_model.eval()

transform = transforms.Compose([
    transforms.Resize(size=(cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.means,std=cfg.std)
])

def predictImage():
    img_list = os.listdir(root)

    for img_name in img_list:
        start_time = time.time()
        img = Image.open(os.path.join(root, img_name))
        w, h = img.size

        image = transform(img)
        x = image.unsqueeze(0)
        x = x.to(device)

        with torch.no_grad():
            detections = eval_model(x, 'test')
        print('detections:', detections.size()) # TODO [1, 21, 200, 5]

        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        #TODO 遍历每一个类别
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            #TODO 过滤掉哪些低置信度的框
            mask = dets[:,0].gt(cfg.conf_thresh).expand(5,dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1,5)
            #TODO 判断当前的检测结果是否包含框
            if dets.shape[0]==0:
                continue
            if j:
                boxes = dets[:,1:]
                boxes[:,0] *= h
                boxes[:,1] *= w
                boxes[:,2] *= h
                boxes[:,3] *= w
                scores = dets[:,0].cpu().numpy()
                for box, score in zip(boxes,scores):
                    y1,x1,y2,x2 = box
                    y1 = int(y1)
                    x1 = int(x1)
                    y2 = int(y2)
                    x2 = int(x2)

                    cv2.rectangle(img=cv_img, pt1=(x1, y1), pt2=(x2, y2),
                                  color=(255, 0, 255), thickness=2, lineType=2)
                    # cv2.putText(cv_img, cfg.VOC_CLASSES[int(j)]+"%.2f"%score, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    #             (255, 0, 255))

                    text = "{} {}%".format(cfg.VOC_CLASSES[int(j)], round(score.item() * 100, 2))
                    cvzone.putTextRect(
                        img=cv_img, text=text, pos=(x1 + 9, y1 - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
                        font=cv2.FONT_HERSHEY_SIMPLEX
                    )
        end_time = time.time()
        print('{} inference time: {} seconds'.format(img_name,end_time - start_time))
        cv2.imwrite(os.path.join(save,img_name),cv_img)
        # cv2.imshow('img', img)
        # k = cv2.waitKey(0)
        # if k & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break


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
        # TODO 输出定位框和score分数
        detections = eval_model(x,'test')
        FPS = int(count / (time.time() - start_time))

        cv_img = frame
        # TODO 遍历每一个类别
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            # TODO 过滤掉哪些低置信度的框
            mask = dets[:, 0].gt(cfg.conf_thresh).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            # TODO 判断当前的检测结果是否包含框
            if dets.shape[0] == 0:
                continue
            if j:
                boxes = dets[:, 1:]
                boxes[:, 0] *= h
                boxes[:, 1] *= w
                boxes[:, 2] *= h
                boxes[:, 3] *= w
                scores = dets[:, 0].cpu().numpy()
                for box, score in zip(boxes, scores):
                    y1, x1, y2, x2 = box
                    y1 = int(y1)
                    x1 = int(x1)
                    y2 = int(y2)
                    x2 = int(x2)

                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
                    cv2.putText(img=cv_img, text=str(int(FPS)), org=(50, 50),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                                color=(0, 255, 0),thickness=2)

                    text = "{} {}%".format(cfg.VOC_CLASSES[int(j)], round(score.item() * 100, 2))
                    cvzone.putTextRect(
                        img=cv_img, text=text, pos=(x1 + 9, y1 - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
                        font=cv2.FONT_HERSHEY_SIMPLEX
                    )
        cv2.imshow('img',cv_img)
        key = cv2.waitKey(1)
        if key & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # predictImage()
    timeDetect()
    pass