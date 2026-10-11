

import os
import torch
import argparse

import cv2
import time
import cvzone
import numpy as np
from PIL import Image
from data.factory import dataset_factory,detection_collate

from data.config import cfg
from s3fd import build_s3fd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.augmentations import to_chw_bgr


parser = argparse.ArgumentParser(description='s3df demo')
parser.add_argument('--save_dir', type=str, default='./outputs/',
                    help='Directory for detect result')
parser.add_argument('--model', type=str,
                    default='weights/sfd_face_135000.pth', help='trained model')
parser.add_argument('--thresh', default=0.5, type=float,
                    help='Final confidence threshold')
parser.add_argument('--data_dir',
                    type=str,default=r'./images',
                    help='Final confidence threshold')
args = parser.parse_args()


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

def detect(net, img_path, thresh):
    img = cv2.imread(img_path)
    height,width,_ = img.shape
    # max_im_shrink = np.sqrt(1700 * 1200 / (img.shape[0] * img.shape[1]))
    # image = cv2.resize(img, None, None, fx=max_im_shrink,
    #                   fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    image = cv2.resize(img, (640, 640))
    #TODO 将图像转换为BGR
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    #TODO 转换为RGB格式
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    x = x.to(device)
    t1 = time.time()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height,width, height])

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print('detection.shape: {}'.format(detections.size()))

    #TODO 对每一个类别进行遍历，其中只有人脸 + 背景（label= 0）
    for i in range(detections.size(1)):
        j = 0
        #TODO 只选择那些大于指定阈值的预测框
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            #TODO 得到当前预测的人脸坐标框位置，同时将坐标值缩放值相对当前图像大小
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            xmin = int(pt[0])
            ymin = int(pt[1])
            xmax = int(pt[2])
            ymax = int(pt[3])
            j += 1
            cv2.rectangle(img,(xmin,ymin) , (xmax,ymax), (255, 0, 255), 2)
            conf = "{:.3f}".format(score)

            cvzone.putTextRect(
                img=img, text=conf, pos=(xmin + 9, ymin - 12),
                scale=0.5, thickness=1, colorR=(0, 255, 0),
                font=cv2.FONT_HERSHEY_SIMPLEX
            )

    t2 = time.time()
    print('detect:{} timer:{}'.format(os.path.basename(img_path), t2 - t1))

    cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)


def timeDetect(net,thresh):

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
        height, width = np.shape(frame)[:2]

        image = cv2.resize(frame, (640, 640))
        # TODO 将图像转换为BGR
        x = to_chw_bgr(image)
        x = x.astype('float32')
        x -= cfg.img_mean
        # TODO 转换为RGB格式
        x = x[[2, 1, 0], :, :]

        x = Variable(torch.from_numpy(x).unsqueeze(0))
        x = x.to(device)
        t1 = time.time()
        with torch.no_grad():
            y = net(x)
        detections = y.data
        scale = torch.Tensor([width, height, width, height])

        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print('detection.shape: {}'.format(detections.size()))

        count += 1

        FPS = count / (time.time() - start_time)

        # TODO 对每一个类别进行遍历，其中只有人脸 + 背景（label= 0）
        for i in range(detections.size(1)):
            j = 0
            # TODO 只选择那些大于指定阈值的预测框
            while detections[0, i, j, 0] >= thresh:
                score = detections[0, i, j, 0]
                # TODO 得到当前预测的人脸坐标框位置，同时将坐标值缩放值相对当前图像大小
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                xmin = int(pt[0])
                ymin = int(pt[1])
                xmax = int(pt[2])
                ymax = int(pt[3])
                j += 1
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)
                cv2.putText(img=frame, text=str(int(FPS)), org=(50, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                            color=(0, 255, 0), thickness=2)

                conf = "{:.3f}".format(score)
                cvzone.putTextRect(
                    img=frame, text=conf, pos=(xmin + 9, ymin - 12),
                    scale=0.5, thickness=1, colorR=(0, 255, 0),
                    font=cv2.FONT_HERSHEY_SIMPLEX
                )
        cv2.imshow('img',frame)
        key = cv2.waitKey(1)
        if key & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def begin():
    net = build_s3fd('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model,map_location='cpu'))
    net.eval()
    net.to(device)

    img_path = args.data_dir
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('jpg')]
    for path in img_list:
        detect(net, path, args.thresh)

def demo():
    train_dataset, val_dataset = dataset_factory('face')

    train_loader = DataLoader(train_dataset, 2,
                              num_workers=8,
                              shuffle=True,
                              collate_fn=detection_collate,
                              pin_memory=True)

    for step,(images,target) in enumerate(train_loader):
        print('image.shape: {}'.format(images.size()))

if __name__ == '__main__':
    # demo()
    begin()

    net = build_s3fd('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model,map_location='cpu'))
    net.eval()
    net.to(device)
    timeDetect(net,args.thresh)
    pass