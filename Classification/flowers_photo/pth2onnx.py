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
from dataset_v2 import loadTransform

import onnx
import onnxruntime

import torch
from torch import nn

def loadModel():
    # TODO 加载模型
    model = Model(in_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    weight_path = r'D:\conda3\Transfer_Learning\B站\day1\video_codes\output\best_0.793.pth'
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
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
    Model = Model.eval().to(config.DEVICE)
    x = torch.rand(size = (1,3,224,224)).to(config.DEVICE)
    #TODO pth => onnx
    with torch.no_grad():
        torch.onnx.export(
            Model,
            x,
            'onnx/best_0.793.onnx',  # 导出的ONNX名称
            opset_version=11,  # ONNX算子集版本
            input_names=['input'],  # 输入Tensor的名称（名称自己决定）
            output_names=['predictions']  # 输出tensor的名称（名称自己决定）
        )
    #TODO 检验当前是否转换成功
    checkONNX(onnxPath='onnx/best_0.793.onnx')

def predict_imgs():
    transform = loadTransform()
    root = r'D:\conda3\Transfer_Learning\B站\day1\codes\images'
    #TODO 加载onnx模型
    onnxModel = onnxruntime.InferenceSession(path_or_bytes=r'onnx/best_0.793.onnx')

    for imgName in os.listdir(root):
        img_path = os.path.join(root,imgName)
        image = Image.open(img_path)
        img = transform(image).unsqueeze(dim=0).to(config.DEVICE).cpu().detach().numpy()#TODO 这里一定要注意转换为numpy类型

        input = {'input':img}
        output = onnxModel.run(output_names=['predictions'],input_feed=input)
        prediction_index = np.argmax(output,axis=-1)[0][0]
        print('gt class: {}  predict class: {}  {:.3f}  name: {}'.format(os.path.basename(imgName),
            config.index_map_name[prediction_index],
            output[0][prediction_index],config.name[prediction_index]
        ))

if __name__ == '__main__':
    # loadONNXClassify()
    predict_imgs()
    pass