"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/11/27-17:54
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import argparse
import time

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
# from utils.nms_wrapper import nms
from torchvision.ops import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer

parser = argparse.ArgumentParser(description='FaceBoxes')

parser.add_argument('-m', '--trained_model',
                    default='weights/FaceBoxes_epoch_30.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='PASCAL', type=str, choices=['AFW', 'PASCAL', 'FDDB'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.5, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--show_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def loadModel(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def predict():

    root = r'./images'
    save = r'./outputs'
    for imgName in os.listdir(root):
        start_time = time.time()

        img_path = os.path.join(root,imgName)
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_raw = cv2.resize(img_raw,dsize=(1024,1024))
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        loc, conf = net(img)  # forward pass
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        if isinstance(boxes,np.ndarray):
            boxes = torch.tensor(boxes,dtype=torch.float32)
        if isinstance(scores,np.ndarray):
            scores = torch.tensor(scores,dtype=torch.float32)

        # keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(boxes=boxes,scores=scores, iou_threshold=args.nms_threshold)
        # do NMS
        boxes = boxes[keep]
        scores = scores[keep]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        cv2.imwrite(os.path.join(save,imgName),img_raw)
        print('{} inference time is :{}'.format(imgName,time.time() - start_time))

    cv2.destroyAllWindows()

def timeDetect():
    cap = cv2.VideoCapture(0)
    count = 0
    start_time = time.time()

    priorbox = PriorBox(cfg, image_size=(1024, 1024))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    while cap.isOpened():
        ret, frame = cap.read()
        count += 1
        if ret == False:
            break

        frame = cv2.resize(frame, dsize=(1024, 1024))
        frame = cv2.flip(src=frame, flipCode=2)
        h, w = np.shape(frame)[:2]

        img = np.float32(frame)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        loc, conf = net(img)  # forward pass

        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        if isinstance(boxes, np.ndarray):
            boxes = torch.tensor(boxes, dtype=torch.float32)
        if isinstance(scores, np.ndarray):
            scores = torch.tensor(scores, dtype=torch.float32)

        # keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(boxes=boxes, scores=scores, iou_threshold=args.nms_threshold)
        # do NMS
        boxes = boxes[keep]
        scores = scores[keep]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(frame, f"FPS {str(int(count / (time.time() - start_time)))}", (50, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0))
            cv2.putText(frame, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255))

        cv2.imshow('img',frame)
        key = cv2.waitKey(1)
        if key&0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # net and model
    pretrained_path = r'./weights/FaceBoxes.pth'
    net = FaceBoxes(phase='test', size=None, num_classes=2)  # initialize detector
    net = loadModel(net, pretrained_path)
    net.eval()
    net.to(device)

    # predict()
    timeDetect()
    pass


