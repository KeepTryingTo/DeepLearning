

import os
import torch
import argparse

import cv2
import time
# import cvzone
import numpy as np
from PIL import Image
from data.factory import dataset_factory,detection_collate

from data.config import cfg
from models.pyramid import build_sfd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from layers.functions.prior_box import PriorBoxLayer


parser = argparse.ArgumentParser(description='s3df demo')
parser.add_argument('--save_dir', type=str, default='./outputs/',
                    help='Directory for detect result')
parser.add_argument('--model', type=str,
                    default=None, help='trained model')
parser.add_argument('--thresh', default=0.20, type=float,
                    help='Final confidence threshold')
parser.add_argument('--data_dir',
                    type=str,default=r'./images',
                    help='Final confidence threshold')
args = parser.parse_args()


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = None
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets

def detect(net, img_path, thresh):
    img = cv2.imread(img_path)
    height,width,_ = img.shape
    # max_im_shrink = np.sqrt(1700 * 1200 / (img.shape[0] * img.shape[1]))
    # image = cv2.resize(img, None, None, fx=max_im_shrink,
    #                   fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    image = cv2.resize(img, dsize = (640, 640))
    #TODO 将图像转换为BGR
    x = image.astype(np.float32)
    x -= np.array([104, 117, 123], dtype=np.float32)

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)

    x = x.to(device)
    t1 = time.time()
    with torch.no_grad():
        net.priorbox = PriorBoxLayer(width = 640, height = 640)
        y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])

    boxes = []
    scores = []
    #TODO 遍历每一个类别
    for i in range(detections.size(1)):
        j = 0
        #TODO 判断当前预测的阈值是否大于指定的阈值
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            boxes.append([pt[0], pt[1], pt[2], pt[3]])
            scores.append(score)
            j += 1
            if j >= detections.size(2):
                break

    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0, 0, 0, 0, 0.001]])

    det_xmin = boxes[:, 0]
    det_ymin = boxes[:, 1]
    det_xmax = boxes[:, 2]
    det_ymax = boxes[:, 3]
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]

    # det = bbox_vote(det)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print('det.shape: {}'.format(np.shape(det)))

    #TODO 对每一个类别进行遍历，其中只有人脸 + 背景（label= 0）
    for i in range(np.shape(det)[0]):
        #TODO 只选择那些大于指定阈值的预测框
        if det[i,4] >= thresh:
            score = det[i,4]
            #TODO 得到当前预测的人脸坐标框位置，同时将坐标值缩放值相对当前图像大小
            pt = det[i,:4]
            xmin = int(pt[0])
            ymin = int(pt[1])
            xmax = int(pt[2])
            ymax = int(pt[3])
            cv2.rectangle(img,(xmin,ymin) , (xmax,ymax), (255, 0, 255), 2)
            conf = "{:.3f}".format(score)

            # cvzone.putTextRect(
            #     img=img, text=conf, pos=(xmin + 9, ymin - 12),
            #     scale=0.5, thickness=1, colorR=(0, 255, 0),
            #     font=cv2.FONT_HERSHEY_SIMPLEX
            # )

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
        x = image.astype(np.float32)
        x -= np.array([104, 117, 123], dtype=np.float32)
        x = x.astype('float32')
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = x.unsqueeze(0)

        x = x.to(device)
        t1 = time.time()
        net.priorbox = PriorBoxLayer(width, height)
        y = net(x)
        detections = y.data
        scale = torch.Tensor([width, height, width, height])

        boxes = []
        scores = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.01:
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                boxes.append([pt[0], pt[1], pt[2], pt[3]])
                scores.append(score)
                j += 1
                if j >= detections.size(2):
                    break

        det_conf = np.array(scores)
        boxes = np.array(boxes)

        if boxes.shape[0] == 0:
            return np.array([[0, 0, 0, 0, 0.001]])

        det_xmin = boxes[:, 0]
        det_ymin = boxes[:, 1]
        det_xmax = boxes[:, 2]
        det_ymax = boxes[:, 3]
        det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

        keep_index = np.where(det[:, 4] >= 0)[0]
        det = det[keep_index, :]
        # det = bbox_vote(det)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print('det.shape: {}'.format(det.size()))

        count += 1
        FPS = count / (time.time() - start_time)

        # TODO 对每一个类别进行遍历，其中只有人脸 + 背景（label= 0）
        for i in range(np.shape(det)[0]):
            j = 0
            # TODO 只选择那些大于指定阈值的预测框
            if det[i,4] >= thresh:
                score = det[i, 4]
                # TODO 得到当前预测的人脸坐标框位置，同时将坐标值缩放值相对当前图像大小
                pt = det[i, :4]
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

                # cvzone.putTextRect(
                #     img=img, text=conf, pos=(xmin + 9, ymin - 12),
                #     scale=0.5, thickness=1, colorR=(0, 255, 0),
                #     font=cv2.FONT_HERSHEY_SIMPLEX
                # )

        cv2.imshow('img',frame)
        key = cv2.waitKey(1)
        if key & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def begin():
    img_path = args.data_dir
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('jpg')]
    for path in img_list:
        detect(net, path, args.thresh)

def demo():
    train_dataset, val_dataset = dataset_factory('face')

    train_loader = DataLoader(train_dataset, batch_size=2,
                              num_workers=8,
                              shuffle=True,
                              collate_fn=detection_collate,
                              pin_memory=True)

    for step,(images,target) in enumerate(train_loader):
        print('image.shape: {}'.format(images.size()))

if __name__ == '__main__':
    # demo()
    net = build_sfd('test', size = 640, num_classes=2)
    checkpoint = torch.load(r'./weights/Res50_pyramid_305000.pth',map_location='cpu')
    net.load_state_dict(checkpoint)
    net.eval().to(device)
    print('load mode is done ...')
    begin()
    # timeDetect(net,thresh=0.5)
    pass