"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/1/4-21:08
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode,Visualizer
from detectron2 import model_zoo

import os
import cv2
import time
import argparse
import numpy as np
from prettytable import PrettyTable

root_config = r'D:/conda3/Transfer_Learning/ObjectDetect/detectron2-main'

class Detector:
    def __init__(
            self,
            save_path,
            model_type = 'OD',
            device = 'cuda',
            is_show = False):
        self.cfg = get_cfg()
        if model_type == 'OD':
            self.cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == 'IS':
            self.cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type == 'KP':
            self.cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                    "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        elif model_type == 'LVIS':
            self.cfg.merge_from_file(model_zoo.get_config_file(
                "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        elif model_type == 'PS':
            self.cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-PanopticSegmentation/panoptic_FPN_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-PanopticSegmentation/panoptic_FPN_R_101_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = device

        self.predictor = DefaultPredictor(self.cfg)
        self.save_path = save_path
        self.is_show = is_show
        self.model_type = model_type

    def predictImage(self,imagePath):
        image = cv2.imread(imagePath)
        if self.model_type != 'PS':
            predictions = self.predictor(image)

            if self.model_type == 'OD':
                viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                 instance_mode=ColorMode.IMAGE)
            elif self.model_type == 'IS':
                viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                 instance_mode=ColorMode.SEGMENTATION)
            elif self.model_type == 'KP':
                viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                 instance_mode=ColorMode.IMAGE)
            elif self.model_type == 'LVIS':
                viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                 instance_mode=ColorMode.SEGMENTATION)

            output = viz.draw_instance_predictions(predictions['instances'].to('cpu'))
        elif self.model_type == 'PS':
            predictions,segmentInfo = self.predictor(image)['panoptic_seg']
            viz = Visualizer(image[:,:,::-1],MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_panoptic_seg_predictions(predictions.to('cpu'), segmentInfo)


        cv2.imwrite(os.path.join(self.save_path,os.path.basename(imagePath)),output.get_image()[:,:,::-1])
        if self.is_show:
            cv2.imshow('img',output.get_image()[:,:,::-1])
        cv2.waitKey(0)

    def predictVideo(self,videoPath):
        cap = cv2.VideoCapture(videoPath)

        if cap.isOpened() == False:
            print('Error opening the video ...')
            return

        while cap.isOpened():
            succes,frame = cap.read()
            if self.model_type != 'PS':
                predictions = self.predictor(frame)

                if self.model_type == 'OD':
                    viz = Visualizer(frame[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                     instance_mode=ColorMode.IMAGE_BW)
                elif self.model_type == 'IS':
                    viz = Visualizer(frame[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                     instance_mode=ColorMode.SEGMENTATION)
                elif self.model_type == 'KP':
                    viz = Visualizer(frame[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                     instance_mode=ColorMode.IMAGE)

                output = viz.draw_instance_predictions(predictions['instances'].to('cpu'))
            elif self.model_type == 'PS':
                predictions, segmentInfo = self.predictor(frame)['panoptic_seg']
                viz = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output = viz.draw_panoptic_seg_predictions(predictions.to('cpu'), segmentInfo)

            cv2.imshow('img', output.get_image()[:, :, ::-1])
            key = cv2.waitKey(1)
            if key&0xff == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def predict(args):
    detector = Detector(save_path=args.save_path,
                        model_type = args.model_type,
                        device = args.device,
                        is_show = args.is_show)
    args.dir = os.path.join(root_config, args.dir)
    args.save_path = os.path.join(root_config, args.save_path)
    if args.demo == 'image':
        for imgName in os.listdir(args.dir):
            start_time = time.time()

            img_path = os.path.join(args.dir,imgName)
            detector.predictImage(img_path)

            print('{} inference time is {}'.format(imgName,time.time() - start_time))
    elif args.demo == 'video':
        detector.predictVideo(args.dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Detectron2 Predictor")
    # ==================================data section==============================================
    parser.add_argument('--model_type', type=str, default='IS',choices=['OD','PS','IS','KP'])
    parser.add_argument('--demo', type=str, default='image',choices=['image','video'])
    parser.add_argument('--dir', type=str, default='./images/',help='if you select the image mode or video mode and the file path')
    parser.add_argument('--save_path', type=str, default=r'./outputs/')
    parser.add_argument('--is_show', type=bool, default=False, help="you want to show image by opencv")
    parser.add_argument('--device', type=str, default='cuda', help="you want to show image by opencv")

    args = parser.parse_args()
    table = PrettyTable()
    table.field_names = ['Argument', 'Value']
    for arg, value in vars(args).items():
        table.add_row([arg, value])
    print('-----------------------argument---------------------------')
    print(table)
    print('----------------------------------------------------------')

    predict(args)