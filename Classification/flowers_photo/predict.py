"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/8/3-19:51
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os

import torch
from torch import nn
from PIL import Image

import config
from model import Model
from dataset_v2 import loadTransform

#TODO 加载模型
model = Model(in_channels=config.IN_CHANNELS,num_classes=config.NUM_CLASSES).to(config.DEVICE)
weight_path = r'C:\TransferLearning\B站\day1\video_codes\output\best_0.793.pth'
checkpoint = torch.load(weight_path,map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

transform = loadTransform()

img_root = r'C:\TransferLearning\B站\day1\video_codes\images'
def predict_imgs():
    for imgName in os.listdir(img_root):
        img_path = os.path.join(img_root,imgName)
        image = Image.open(img_path)
        image = transform(image).unsqueeze(dim = 0).to(config.DEVICE) # [3,224,224] => [1,3,224,224]

        output = model(image) #[1,10]
        prediction_index = torch.argmax(output,dim = -1).item()
        print('predict class {}   {:.3f}  name: {}'.format(
            config.index_map_name[prediction_index],output[0][prediction_index],
            config.name[prediction_index]
        ))

if __name__ == '__main__':
    predict_imgs()
    pass