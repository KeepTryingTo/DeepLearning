"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/9/25-10:39
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import os
from PIL import Image
import config

import torch
from torchvision import models
from torch.utils.mobile_optimizer import optimize_for_mobile

from dataset_v2 import loadTransform
from model_finetune import modelFineTune

def loadModel():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT,progress=True)
    model.eval()
    return model

def loadCustomModel():
    # TODO 加载模型
    model = modelFineTune(num_classes=config.NUM_CLASSES,pretrained=config.PRETRAINED,
                          isFreezeBackbone=config.ISFREEZEBACKBONE,
                          freeze_layers=config.FREEZE_LAYERS,model_name='mobilenetv3')
    weight_path = r'D:\conda3\Transfer_Learning\B Stand\day1\video_codes\output\best_layers_5_finetune_0.901_15.pth'
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def pth2pt_f():
    # model = loadModel()
    model = loadCustomModel()
    x = torch.zeros(size=(1,3,224,224))
    jit_model = torch.jit.trace(model,x)
    optimize_jit_model = optimize_for_mobile(jit_model)
    optimize_jit_model._save_for_lite_interpreter("./onnx/custom_model.pt")

def loadpt_and_predict():
    optimize_jit_model = torch.jit.load("./onnx/custom_model.pt")

    transform = loadTransform()
    with open("imageNet_classes.txt",'r',encoding='utf-8') as fp:
        lines = fp.readlines()

    classes = []
    for line in lines:
        classes.append(line.strip("\n"))

    img_root = r'D:\conda3\Transfer_Learning\B stand\day1\video_codes\images'

    img_list = os.listdir(img_root)
    img_list = sorted(sorted(img_list, key=lambda x: (len(x), x), reverse=False))
    for imgName in img_list:
        img_path = os.path.join(img_root, imgName)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(dim=0)  # [3,224,224] => [1,3,224,224]

        output = optimize_jit_model(image)  # [1,5]
        prediction_index = torch.argmax(output, dim=-1).item()
        print('ground truth class {}  predict class {}   {:.3f}  name: {}'.format(imgName.split('.')[0],
                                                                                      config.index_map_name[
                                                                                          prediction_index],
                                                                                      output[0][prediction_index],
                                                                                      config.name[prediction_index]
                                                                                      ))
        # print("predict class: {}".format(classes[prediction_index]))
if __name__ == '__main__':
    # pth2pt_f()
    loadpt_and_predict()
    pass


