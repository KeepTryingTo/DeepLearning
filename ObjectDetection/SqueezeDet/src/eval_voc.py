#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import torch
import pickle
import argparse
from utils.timer import Timer
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from utils.core import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class_names = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


from utils.config import Config
from utils.misc import init_env
from engine.detector import Detector
from model.squeezedet import SqueezeDet
from utils.model import load_model





num_classes = len(class_names)


def detect(img,model,detector):
    dict_image = {'image': img}
    with torch.no_grad():
        dets = model(dict_image)
        batch_size = dets['class_ids'].shape[0]
        for b in range(batch_size):
            det = {k: v[b] for k, v in dets.items()}
            det = detector.filter(det)

            if det is None:
                continue

            det = {k: v.cpu().numpy() for k, v in det.items()}

    return det

def boxes_labels_process(num_classes, i, scores, boxes, min_thresh, all_boxes, max_per_image):
    for j in range(num_classes):  # ignore the bg(category_id=0)
        inds = np.where(scores[j] > min_thresh)[0]
        if len(inds) == 0:
            all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]

        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)

        all_boxes[j][i] = c_dets
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1]
                                  for j in range(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]

def test_net(model,detector,save_folder, testset,
             transform, max_per_image=300, thresh=0.005):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    num_images = len(testset)
    print_info('=> Total {} images to test.'.format(
        num_images), ['yellow', 'bold'])
    all_boxes = [[[] for _ in range(num_images)] for _ in range(len(testset.class_names))]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')
    tot_detect_time, tot_nms_time = 0, 0
    print_info('Begin to evaluate', ['yellow', 'bold'])

    for i in tqdm(range(num_images)):
        img = testset.pull_image(i)
        # step1: CNN detection
        _t['im_detect'].tic()
        x = transform(img).unsqueeze(0).cuda()
        det = detect(x,model,detector=detector)
        if det is None:
            continue
        class_ids, boxes, scores = det['class_ids'], det['boxes'], det['scores']
        # print('boxes.shape: {}'.format(np.shape(boxes)))
        # print('scores.shape: {}'.format(np.shape(scores)))
        detect_time = _t['im_detect'].toc()
        # step2: Post-process: NMS
        _t['misc'].tic()
        nms_time = _t['misc'].toc()

        boxes_labels_process(num_classes=num_classes,i = i,
                             scores=scores,boxes=boxes,
                             min_thresh=0.005,all_boxes=all_boxes,
                             max_per_image=max_per_image)

        tot_detect_time += detect_time if i > 0 else 0
        tot_nms_time += nms_time if i > 0 else 0

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print_info('===> Evaluating detections', ['yellow', 'bold'])
    map = testset.evaluate_detections(all_boxes, save_folder)
    print_info('Detect time per image: {:.3f}s'.format(
        tot_detect_time / (num_images - 1)))
    print_info('Nms time per image: {:.3f}s'.format(
        tot_nms_time / (num_images - 1)))
    print_info('Total time per image: {:.3f}s'.format(
        (tot_detect_time + tot_nms_time) / (num_images - 1)))
    print_info('FPS: {:.3f} fps'.format(
        (num_images - 1) / (tot_detect_time + tot_nms_time)))

    return map


def loadVOCDataset():
    from src.datasets.voc0712 import VOCDetection
    from src.datasets.data_augment import BaseTransform
    root_dir = r'/data1/KTG/myDataset/VOC'
    VOC = dict(
        train_sets=[('2007', 'trainval'), ('2012', 'trainval')],
        eval_sets=[('2007', 'test')],
    )
    img_size = (512,768)
    rgb_means = (103.94, 116.78, 123.68)

    _preprocess = BaseTransform(
        resizes=img_size,
        rgb_means=rgb_means,
        swap = (2, 0, 1))

    valDataset = VOCDetection(
        img_size=img_size,
        root=root_dir, image_sets=VOC['eval_sets']
    )

    return valDataset,_preprocess


def load_model(model, model_state_dict):
    state_dict_ = model_state_dict
    state_dict = {}
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    model_state_dict = model.state_dict()
    # check loaded parameters and created model parameters
    success_loaded = True
    for layer in state_dict:
        if layer in model_state_dict:
            if state_dict[layer].shape != model_state_dict[layer].shape:
                success_loaded = False
                print('Skip loading param {}, required shape{}, loaded shape{}.'.format(
                    layer, model_state_dict[layer].shape, state_dict[layer].shape))
                state_dict[layer] = model_state_dict[layer]
        else:
            success_loaded = False
            print('Drop param {} in pre-trained model.'.format(layer))

    for layer in model_state_dict:
        if layer not in state_dict:
            success_loaded = False
            print('Param {} not found in pre-trained model.'.format(layer))
            state_dict[layer] = model_state_dict[layer]

    model.load_state_dict(state_dict, strict=False)
    print('Model successfully loaded.' if success_loaded else
          'The model does not fully load the pre-trained weight.')

    return model

def eval(model_state_dict):
    save_folder = r'/data1/KTG/myProject/SqueezeDet-PyTorch-master/exp/save'

    cfg = Config().parse()
    init_env(cfg)
    testset, _preprocess = loadVOCDataset()
    cfg = Config().update_dataset_info(cfg = cfg, dataset=testset)

    # prepare model & detector
    model = SqueezeDet(cfg)
    model = load_model(model, model_state_dict)
    detector = Detector(model, cfg)


    map = test_net(model = model,detector = detector,
             save_folder=save_folder,
             testset=testset,
             transform=_preprocess,
             max_per_image=300,
             thresh=0.005)

    return map


if __name__ == '__main__':
    weight_path = r''
    save_folder = r'/data1/KTG/myProject/SqueezeDet-PyTorch-master/exp/save'
    testset, _preprocess = loadVOCDataset()
    cfg = Config().parse()
    init_env(cfg)
    cfg = Config().update_dataset_info(cfg=cfg, dataset=testset)

    model = SqueezeDet(cfg)
    model = load_model(model, weight_path)
    detector = Detector(model, cfg)

    map = test_net(model=model,
                   detector=detector,
                   save_folder=save_folder,
                   testset=testset,
                   transform=_preprocess,
                   max_per_image=100,
                   thresh=0.005)
