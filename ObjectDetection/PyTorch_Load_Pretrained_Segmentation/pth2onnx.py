"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/8/16-7:29
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import time

import cv2

import config
import numpy as np
from PIL import Image

import onnx
import onnxruntime

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

def checkONNX(onnxPath='onnx/fcn_resnet50.onnx'):
    #加载ONNX模型
    modelONNX = onnx.load(onnxPath)
    #检查模型格式是否正确
    onnx.checker.check_model(modelONNX)
    #打印ONNX的计算图
    # print(onnx.helper.printable_graph(modelONNX.graph))
    return modelONNX

def loadONNXObject():
    model = models.segmentation.fcn_resnet50(pretrained=True,
                                            progress=True).to(config.device)
    model.eval().to(config.device)
    x = torch.rand(size = (1,3,config.img_size,config.img_size)).to(config.device)
    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            'onnx/fcn_resnet50.onnx',  # 导出的ONNX名称
            do_constant_folding=True,#do_constant_folding=True 使静态图优化更彻底。
            opset_version=11,  # ONNX算子集版本
            input_names=['input'],  # 输入Tensor的名称（名称自己决定）
            output_names=['out','aux']  # 输出tensor的名称（名称自己决定）
        )
    checkONNX(onnxPath='onnx/fcn_resnet50.onnx')

def segmentImage():
    onnxFcn = onnxruntime.InferenceSession('onnx/fcn_resnet50.onnx')
    color_map = config.create_color_map(num_classes=len(config.palette))

    images_list = os.listdir(config.root)
    for imgName in images_list:
        starTime = time.time()
        img_path = os.path.join(config.root, imgName)
        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert("RGB")
        image = config.transform(img).unsqueeze(dim=0).numpy()

        with torch.no_grad():
            out,aux = onnxFcn.run(output_names=['out','aux'],input_feed={'input':image})
        out = out[0]
        aux = aux[0]
        endTime = time.time()
        print('out.shape: {}'.format(np.shape(out)))
        print('aux.shape: {}'.format(np.shape(aux)))

        prediction = np.squeeze(out)
        prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
        # TODO 可视化方式一 =================================================
        colorized_mask = config.save_images(image=img, mask=prediction, output_path=config.save_dir, image_file=imgName,
                                            palette=config.palette, num_classes=len(config.palette))
        # colorized_mask.show()
        # TODO # 可视化方式二 =================================================
        # color_image = config.output_to_color_image(prediction,color_map)
        # color_image = Image.fromarray(color_image)
        # color_image.save(os.path.join(config.save_dir,imgName))
        # color_image.show()

        print('segment {} time is {}s'.format(imgName, endTime - starTime))

def timeSegmentImage():
    onnxFcn = onnxruntime.InferenceSession('onnx/fcn_resnet50.onnx')
    cap = cv2.VideoCapture(0)
    color_map = config.create_color_map(len(config.palette))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break

        img = cv2.resize(frame, dsize=(config.img_size, config.img_size))
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = config.transform(image).unsqueeze(dim=0).numpy()

        with torch.no_grad():
            out, aux = onnxFcn.run(output_names=['out', 'aux'], input_feed={'input': image})
        out = out[0]
        aux = aux[0]

        print('out.shape: {}'.format(np.shape(out)))
        print('aux.shape: {}'.format(np.shape(aux)))

        prediction = np.squeeze(out)
        prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

        color_image = config.output_to_color_image(prediction,color_map)
        color_image = Image.fromarray(color_image)

        cv_img = np.array(color_image)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        cv2.imshow('img', cv_img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    loadONNXObject()
    # segmentImage()
    # timeSegmentImage()
    pass