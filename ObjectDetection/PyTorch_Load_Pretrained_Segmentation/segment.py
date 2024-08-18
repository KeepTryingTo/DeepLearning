"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/8/18-10:30
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import cv2
import time

import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

import config

def loadModel():
    model = models.segmentation.fcn_resnet50(pretrained=True,
                                             progress=True).to(config.device)
    model.eval()
    return model

def segmentImage():
    model = loadModel()
    color_map = config.create_color_map(num_classes=len(config.palette))
    images_list = os.listdir(config.root)
    for imgName in images_list:
        starTime = time.time()
        img_path = os.path.join(config.root,imgName)
        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert("RGB")
        image = config.transform(img).unsqueeze(dim = 0).to(config.device)

        with torch.no_grad():
            output = model(image)

        out = output['out']
        aux = output['aux']
        endTime = time.time()

        print('out.shape: {}'.format(out.shape))
        print('aux.shape: {}'.format(aux.shape))

        prediction = out.squeeze(0).cpu().numpy()
        prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
        print(prediction)
        # TODO 可视化方式一 =================================================
        colorized_mask = config.save_images(image=img, mask=prediction, output_path=config.save_dir, image_file=imgName,
                                     palette=config.palette, num_classes=len(config.palette))
        # colorized_mask.show()
        #TODO # 可视化方式二 =================================================
        # color_image = config.output_to_color_image(prediction,color_map)
        # color_image = Image.fromarray(color_image)
        # color_image.save(os.path.join(config.save_dir,imgName))
        # color_image.show()

        print('segment {} time is {}s'.format(imgName,endTime - starTime))


def timeSegmentImage():
    model = loadModel()
    cap = cv2.VideoCapture(0)
    color_map = config.create_color_map(len(config.palette))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break

        img = cv2.resize(frame,dsize=(config.img_size,config.img_size))
        image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = config.transform(image).unsqueeze(dim=0).to(config.device)

        with torch.no_grad():
            output = model(image)

        out = output['out']
        aux = output['aux']

        print('out.shape: {}'.format(out.shape))
        print('aux.shape: {}'.format(aux.shape))

        prediction = out.squeeze(0).cpu().numpy()
        prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

        color_image = config.output_to_color_image(prediction,color_map)
        color_image = Image.fromarray(color_image)

        cv_img = np.array(color_image)
        cv_img = cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)

        cv2.imshow('img',cv_img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    segmentImage()
    # timeSegmentImage()
    pass


