"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/8/14-12:28
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import config
import numpy as np
from PIL import Image
from model import Model
from model_finetune import modelFineTune
from dataset_v2 import loadTransform

import onnx
import onnxruntime

import torch
from torch import nn
from torchvision import models

def loadModel():
    # TODO 加载模型
    # model = Model(in_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model = modelFineTune(num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED,
                          freeze_layers=config.FREEZE_LAYERS, isFreezeBackbone=config.ISFREEZEBACKBONE,
                          model_name='mobilenetv3').to(config.DEVICE)
    weight_path = r'D:\conda3\Transfer_Learning\B Stand\day1\video_codes\output\best_5_finetune_0.91_5.pth'
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model

def loadPyTorchModel():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT,
                                      progress=True).to(config.DEVICE)
    return model
def checkONNX(onnxPath='onnx/best.onnx'):
    #加载ONNX模型
    modelONNX = onnx.load(onnxPath)
    #检查模型格式是否正确
    onnx.checker.check_model(modelONNX)
    #打印ONNX的计算图
    # print(onnx.helper.printable_graph(modelONNX.graph))
    return modelONNX

def loadONNXClassify():
    Model = loadModel()
    # Model = loadPyTorchModel()
    Model = Model.eval().to(config.DEVICE)
    x = torch.rand(size = (1,3,224,224)).to(config.DEVICE)
    #TODO pth => onnx
    with torch.no_grad():
        torch.onnx.export(
            Model,
            x,
            'onnx/best_5_finetune_0.91_5.onnx',  # 导出的ONNX名称
            opset_version=11,  # ONNX算子集版本
            input_names=['input'],  # 输入Tensor的名称（名称自己决定）
            output_names=['predictions']  # 输出tensor的名称（名称自己决定）
        )
    #TODO 检验当前是否转换成功
    checkONNX(onnxPath='onnx/best_5_finetune_0.91_5.onnx')

def predict_imgs():
    transform = loadTransform()
    root = r'D:\conda3\Transfer_Learning\B Stand\day1\video_codes\images'
    #TODO 加载onnx模型
    onnxModel = onnxruntime.InferenceSession(path_or_bytes=r'onnx/mobilenet_v3_small.onnx')

    #读取类别文件
    with open("imageNet_classes.txt","r",encoding="utf-8") as fp:
        lines = fp.readlines()

    classes = []
    for line in lines:
        classes.append(line.strip("\n"))
    print('classes size: {}'.format(len(classes)))

    for imgName in os.listdir(root):
        img_path = os.path.join(root,imgName)
        image = Image.open(img_path)
        img = transform(image).unsqueeze(dim=0).to(config.DEVICE).cpu().detach().numpy()#TODO 这里一定要注意转换为numpy类型

        input = {'input':img}
        output = onnxModel.run(output_names=['predictions'],input_feed=input)
        prediction_index = np.argmax(output,axis=-1)[0][0]
        print('gt class: {}  predict class: {}  conf: {:.3f}  name: {}'.format(
            os.path.basename(imgName),
            config.index_map_name[prediction_index],
            output[0][0][prediction_index],
            config.name[prediction_index]
        ))
        # print("prediction class: {}".format(classes[prediction_index]))

def lookModelName():
    # Load the ONNX model
    model = onnx.load(r'D:\conda3\Transfer_Learning\B Stand\day1\video_codes\onnx\mobilenet_v3_small.onnx')

    # Print the model's input names
    for input in model.graph.input:
        print('input.name: {}'.format(input.name))
    for output in model.graph.output:
        print('ouput.name: {}'.format(output.name))


if __name__ == '__main__':
    # loadONNXClassify()
    predict_imgs()
    # lookModelName()
    pass