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
import torch
import warnings
warnings.filterwarnings('ignore')
import argparse

import torch
from data import (base_transform, VOC_CLASSES,mb_cfg)
from layers.functions import Detect,PriorBox


parser = argparse.ArgumentParser(description='TDRN Evaluation')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model',
                    default=r'./weights/drn320_VGG16_7934.pth',
                    type=str, help='Trained state_dict file path to open')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('loading model!')
from model.dualrefinedet_vggbn import build_net
net = build_net(phase='test', size=320,
                num_classes=len(VOC_CLASSES) + 1,
                c7_channel=1024, def_groups=1,
                multihead=True, bn=False)
checkpoint = torch.load(args.trained_model,map_location='cpu')
# for key in checkpoint.keys():
#     print(key)
# print('-----------------------------------------')
# for name,param in net.named_parameters():
#     print(name)
net.load_state_dict(checkpoint)
net.eval()
net = net.to(device)
print('Finished loading model!')

prior = 'VOC_'+ str(320)
cfg = mb_cfg[prior]

mean = (104, 117, 123)
confidence_threshold = 0.85
nms_threshold = 0.45
top_k = 200
detector = Detect(len(VOC_CLASSES) + 1,
                  bkg_label=0, top_k=top_k,
                  conf_thresh=confidence_threshold,
                  nms_thresh=nms_threshold)
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward().to(device)
print('Finished loading model!')

root = r'./images/'
save = r'./outputs/'


def predictImage():
    img_list = os.listdir(root)

    for img_name in img_list:
        start_time = time.time()
        image_path = os.path.join(root, img_name)
        image = cv2.imread(image_path, 1)
        h, w, _ = image.shape
        im_trans = base_transform(image, 320, mean)

        x = torch.from_numpy(im_trans).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        #TODO forward
        with torch.no_grad():
            arm_loc,_, loc, conf = net(x)
            detections = detector.forward(loc, conf, priors, arm_loc_data=arm_loc)

        out = list()
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            if dets.sum() == 0:
                continue
            mask = dets[:, 0].gt(0.).expand(dets.size(-1), dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, dets.size(-1))
            boxes = dets[:, 1:-1] if dets.size(-1) == 6 else dets[:, 1:]
            boxes = boxes / 320
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            boxes_np = boxes.cpu().numpy()

            #TODO 遍历每一个类别
            for box,score in zip(boxes_np,scores):
                box = [int(_) for _ in box]
                confidence = score

                if confidence > confidence_threshold:
                    x1,y1,x2,y2 = box
                    y1 = int(y1)
                    x1 = int(x1)
                    y2 = int(y2)
                    x2 = int(x2)
                    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                        continue
                    if x1 > w or x2 > w or y1 > h or y2 > h:
                        continue

                    cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2),
                                  color=(255, 0, 255), thickness=2, lineType=2)
                    # cv2.putText(cv_img, cfg.VOC_CLASSES[int(j)]+"%.2f"%score, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    #             (255, 0, 255))

                    # text = "{} {}%".format(VOC_CLASSES[int(j) - 1], round(score.item() * 100, 2))
                    # cvzone.putTextRect(
                    #     img=cv_img, text=text, pos=(x1 + 9, y1 - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
                    #     font=cv2.FONT_HERSHEY_SIMPLEX
                    # )
        end_time = time.time()
        print('{} inference time: {} seconds'.format(img_name,end_time - start_time))
        cv2.imwrite(os.path.join(save,img_name),image)


def timeDetect(conf_thresh = 0.5):

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

        h, w, _ = frame.shape
        im_trans = base_transform(frame, 320, mean)

        x = torch.from_numpy(im_trans).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        # TODO forward
        with torch.no_grad():
            arm_loc, _, loc, conf = net(x)
            detections = detector.forward(loc, conf, priors, arm_loc_data=arm_loc)

        out = list()
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            if dets.sum() == 0:
                continue
            mask = dets[:, 0].gt(0.).expand(dets.size(-1), dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, dets.size(-1))
            boxes = dets[:, 1:-1] if dets.size(-1) == 6 else dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            boxes_np = boxes.cpu().numpy()

            # TODO 遍历每一个类别
            for box, score in zip(boxes_np, scores):
                box = [int(_) for _ in box]
                confidence = score

                if confidence > conf_thresh:
                    x1, y1, x2, y2 = box
                    y1 = int(y1)
                    x1 = int(x1)
                    y2 = int(y2)
                    x2 = int(x2)

                    cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2),
                                  color=(255, 0, 255), thickness=2, lineType=2)
                    # cv2.putText(cv_img, cfg.VOC_CLASSES[int(j)]+"%.2f"%score, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    #             (255, 0, 255))

                    # text = "{} {}%".format(VOC_CLASSES[int(j) - 1], round(score.item() * 100, 2))
                    # cvzone.putTextRect(
                    #     img=cv_img, text=text, pos=(x1 + 9, y1 - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
                    #     font=cv2.FONT_HERSHEY_SIMPLEX
                    # )
        cv2.imshow('img',frame)
        key = cv2.waitKey(1)
        if key & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    predictImage()
    # timeDetect()
    pass