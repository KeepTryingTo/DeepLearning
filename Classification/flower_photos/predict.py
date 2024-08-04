"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/8/1-22:36
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import os.path

import torch
import config
from PIL import Image
from torch import nn
from model import Model
from dataset import loadTransform

#TODO 加载模型
model = Model(in_channel=config.IN_CHANNELS, num_classes=config.NUM_CLASSES).to(config.DEVICE)

transform = loadTransform()

root = r'C:\TransferLearning\B站\day1\codes\images'
weight_path = r''
checkpoint = torch.load(weight_path,map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

def predict_imgs():
    for imgName in os.listdir(root):
        img_path = os.path.join(root,imgName)
        image = Image.open(img_path)
        img = transform(image).unsqueeze(dim=0)

        output = model(img)
        prediction = torch.argmax(output,dim=-1).item()
        print('predict class: {}  name: {}'.format(config.index_map_name[prediction],config.name[prediction]))

if __name__ == '__main__':
    predict_imgs()
    pass