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
import torch
import pickle
import argparse
from utils.timer import Timer
import torch.backends.cudnn as cudnn
from layers.functions import Detect, PriorBox
from data import BaseTransform
from configs.CC import Config
from tiny_dsod import Framework
from tqdm import tqdm
from utils.core import *
from data.voc0712 import VOC_CLASSES
from torchvision.ops import nms


parser = argparse.ArgumentParser(description='Tiny DSOD Evaluation')
parser.add_argument(
    '-c', '--config', default='configs/tiny_VOC.py', type=str)
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model',
                    default=r'./weights/VOC/tiny_dsod_VOC_size300_epoch390.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--test', action='store_true',
                    help='to submit a test file')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print_info('----------------------------------------------------------------------\n'
           '|                       Tiny DSOD Evaluation Program                     |\n'
           '----------------------------------------------------------------------', ['yellow', 'bold'])


cfg = Config.fromfile(args.config)
if not os.path.exists(cfg.test_cfg.save_folder):
    os.mkdir(cfg.test_cfg.save_folder)
anchor_config = anchors(cfg.model)
print_info('The Anchor info: \n{}'.format(anchor_config))
priorbox = PriorBox(anchor_config)
with torch.no_grad():
    priors = priorbox.forward()
    if cfg.test_cfg.cuda:
        priors = priors.to(device)

net = Framework(num_class=cfg.model.num_classes)
net = init_net(net,args.trained_model)
print_info('===> Finished constructing and loading model', ['yellow', 'bold'])
net.eval()
net.to(device)
print('Finished loading model!')

num_classes = cfg.model.num_classes
root = r'./images/'
save = r'./outputs/'



_preprocess = BaseTransform(
    cfg.model.input_size,
    cfg.model.rgb_means,
    (2, 0, 1)
)
detector = Detect(num_classes,
                  cfg.loss.bkg_label,
                  anchor_config)


def predictImage(conf_thresh = 0.35):
    img_list = os.listdir(root)

    for img_name in img_list:
        start_time = time.time()
        image = cv2.imread(os.path.join(root,img_name), cv2.IMREAD_COLOR)
        w, h = image.shape[1], image.shape[0]
        img = _preprocess(image).unsqueeze(0)

        if cfg.test_cfg.cuda:
            img = img.to(device)
        scale = torch.Tensor([w, h, w, h]).to(device)
        #TODO forward
        with torch.no_grad():
            out = net(img)
            boxes, scores = detector.forward(out, priors)

        boxes = (boxes[0] * scale).cpu().numpy()
        scores = scores[0].cpu().numpy()
        allboxes = []
        #TODO 对每一个类别进行遍历
        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > cfg.test_cfg.score_threshold)[0]
            if len(inds) == 0:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]

            if isinstance(c_bboxes, np.ndarray):
                c_bboxes = torch.from_numpy(c_bboxes)
            if isinstance(c_scores, np.ndarray):
                c_scores = torch.from_numpy(c_scores)

            keep = nms(c_bboxes, c_scores, iou_threshold=cfg.test_cfg.iou)
            keep = keep[:cfg.test_cfg.keep_per_class]
            c_bboxes = c_bboxes[keep].cpu().numpy()
            c_scores = c_scores[keep].cpu().numpy()

            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            allboxes.extend([_.tolist() + [j] for _ in c_dets])

        allboxes = np.array(allboxes)
        boxes = allboxes[:, :4]
        scores = allboxes[:, 4]
        cls_inds = allboxes[:, 5]

        cv_img = image
        #TODO 遍历每一个类别
        for i,box in enumerate(boxes):
            box = [int(_) for _ in box]
            confidence = scores[i]
            label = cls_inds[i]

            if confidence > conf_thresh:
                x1,y1,x2,y2 = box
                y1 = int(y1)
                x1 = int(x1)
                y2 = int(y2)
                x2 = int(x2)

                cv2.rectangle(img=cv_img, pt1=(x1, y1), pt2=(x2, y2),
                              color=(255, 0, 255), thickness=2, lineType=2)
                # cv2.putText(cv_img, cfg.VOC_CLASSES[int(label)]+"%.2f"%score, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                #             (255, 0, 255))

                # text = "{} {}%".format(VOC_CLASSES[int(label) - 1], round(score.item() * 100, 2))
                # cvzone.putTextRect(
                #     img=cv_img, text=text, pos=(x1 + 9, y1 - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
                #     font=cv2.FONT_HERSHEY_SIMPLEX
                # )
        end_time = time.time()
        print('{} inference time: {} seconds'.format(img_name,end_time - start_time))
        cv2.imwrite(os.path.join(save,img_name),cv_img)


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

        img = _preprocess(frame).unsqueeze(0)
        if cfg.test_cfg.cuda:
            img = img.to(device)
        scale = torch.Tensor([w, h, w, h])
        with torch.no_grad():
            out = net(img)
            boxes, scores = detector.forward(out, priors)

        boxes = (boxes[0] * scale).cpu().numpy()
        scores = scores[0].cpu().numpy()
        allboxes = []
        # TODO 对每一个类别进行遍历
        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > cfg.test_cfg.score_threshold)[0]
            if len(inds) == 0:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]

            if isinstance(c_bboxes, np.ndarray):
                c_bboxes = torch.from_numpy(c_bboxes)
            if isinstance(c_bboxes, np.ndarray):
                c_scores = torch.from_numpy(c_scores)
            keep = nms(c_bboxes, c_scores, iou_threshold=cfg.test_cfg.iou)
            keep = keep[:cfg.test_cfg.keep_per_class]
            c_bboxes = c_bboxes[keep].cpu().numpy()
            c_scores = c_scores[keep].cpu().numpy()

            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            allboxes.extend([_.tolist() + [j] for _ in c_dets])

        allboxes = np.array(allboxes)
        boxes = allboxes[:, :4]
        scores = allboxes[:, 4]
        cls_inds = allboxes[:, 5]

        cv_img = frame
        # TODO 遍历每一个类别
        for i, box in enumerate(boxes):
            box = [int(_) for _ in box]
            confidence = scores[i]
            label = cls_inds[i]

            if confidence > conf_thresh:
                x1, y1, x2, y2 = box
                y1 = int(y1)
                x1 = int(x1)
                y2 = int(y2)
                x2 = int(x2)

                cv2.rectangle(img=cv_img, pt1=(x1, y1), pt2=(x2, y2),
                              color=(255, 0, 255), thickness=2, lineType=2)
                # cv2.putText(cv_img, cfg.VOC_CLASSES[int(label)]+"%.2f"%score, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                #             (255, 0, 255))

                # text = "{} {}%".format(VOC_CLASSES[int(label) - 1], round(score.item() * 100, 2))
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