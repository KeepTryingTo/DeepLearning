from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot,COCOroot 
from data import (AnnotationTransform, VOCDetection,
                  BaseTransform, VOC_300,VOC_512,COCO_300,COCO_512, COCO_mobile_300)

import torch.utils.data as data
from layers.functions import Detect,PriorBox
# from utils.nms_wrapper import nms
from utils.timer import Timer
from torchvision.ops import nms

device = torch.device("cuda:0" if torch.cuda.is_available() else "")

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='myFeature_FSSD_Mobilenet',
                    help='myFeature_FSSD_Vgg or myFeature_FSSD_Mobilenet version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model',
                    default='./runs/myFeature_FSSD_Mobilenet_VOC_epoches_255.pth', #Final_RFB_vgg_VOC.pth
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='./outputs', type=str,
                    help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'myFeature_FSSD_Vgg':
    from models.FSSD_vgg import build_net
elif args.version == 'cfenet':
    from models.cfenet import CFENet
else:
    print('Unkown version!')

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.to(device)


def test_net(save_folder, net,
             detector, cuda,
             testset, transform,
             max_per_image=300,
             iou_thresh=0.5,conf_threshod = 0.2):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    num_classes = (21, 81)[args.dataset == 'COCO']
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return


    #TODO 对所有的图像进行检测
    for i in range(num_images):
        img = testset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if cuda:
                x = x.to(device)
                scale = scale.to(device)

        _t['im_detect'].tic()
        out = net(x,test = True)      # forward pass
        boxes, scores = detector.forward(out,priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]
        boxes *= scale

        _t['misc'].tic()
        #TODO 针对每一个类别都进行遍历
        for j in range(1, num_classes):
            inds = torch.where(scores[:, j] > conf_threshod)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            #TODO 得到当前类别的预测框以及scores分数
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]

            #TODO 这里采用IOU阈值在NMS算法中用于过滤掉重叠的框
            # from utils.box_utils import nms
            # keep,_= nms(c_bboxes, c_scores,
            #              overlap=iou_thresh,
            #              top_k=max_per_image)
            keep = nms(c_bboxes, c_scores,iou_threshold=iou_thresh)

            c_bboxes = c_bboxes[keep]
            c_scores = c_scores[keep]
            c_bboxes = c_bboxes.cpu().numpy()
            c_scores = c_scores.cpu().numpy()

            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)


if __name__ == '__main__':
    # load net
    img_dim = (300,512)[args.size=='512']
    num_classes = (21, 81)[args.dataset == 'COCO']
    # net = build_net(img_dim, num_classes)    # initialize detector
    net = CFENet(num_classes=num_classes)
    state_dict = torch.load(args.trained_model)
    # create new OrderedDict that does not contain `module.`

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    print('Finished loading model!')
    # print(net)
    # load data
    print("voc root: {}".format(VOCroot))
    if args.dataset == 'VOC':
        testset = VOCDetection(
            root=VOCroot,
            # image_sets = [('2012', 'val')],
            image_sets=[('2007', 'test')],
            preproc=None,
            target_transform=AnnotationTransform())
    elif args.dataset == 'COCO':
        # testset = COCODetection(
        #     COCOroot, [('2014', 'minival')], None)
        #     #COCOroot, [('2015', 'test-dev')], None)
        pass
    else:
        print('Only VOC and COCO dataset are supported now!')
    if args.cuda:
        net = net.to(device)
        cudnn.benchmark = True
    else:
        net = net.cpu()
    # evaluation
    #top_k = (300, 200)[args.dataset == 'COCO']
    top_k = 200
    detector = Detect(num_classes,bkg_label=0,cfg=cfg,device=device)
    save_folder = os.path.join(args.save_folder,args.dataset)
    rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']
    test_net(save_folder, net,
             detector, args.cuda,
             testset,
             BaseTransform(net.size, rgb_means, (2, 0, 1)),
             top_k,
             iou_thresh=0.005,conf_threshod = 0.005)

"""
    
('2007', 'trainval'), ('2012', 'trainval') and test on 2007 test.txt for vgg
    iou_thresh=0.005,conf_threshod = 0.005 MAP = 0.7131
    
('2007', 'trainval'), ('2012', 'trainval') and test on 2007 test.txt for mobilenet
    iou_thresh=0.005,conf_threshod = 0.005 MAP = 0.6070  
    0.6103
"""