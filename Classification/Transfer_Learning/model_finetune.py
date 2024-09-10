"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/9/8-9:38
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from torchvision import models


class modelFineTune(nn.Module):
    def __init__(self,num_classes = 5,is_freeze = True,pretrained = True
                 ,model_name = 'mobilenetv3'):
        super().__init__()
        self.num_classes = num_classes
        self.is_freeze = is_freeze
        self.model_name = model_name
        if pretrained:
            if model_name == "mobilenetv3":
                self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT,progress=True)
            elif model_name == "resnet18":
                self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT,progress=True)
            elif model_name == "inceptionv3":
                self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT,progress=True)
        else:
            if model_name == "mobilenetv3":
                self.model = models.mobilenet_v3_small()
            elif model_name == "resnet18":
                self.model = models.resnet18()
            elif model_name == "inceptionv3":
                self.model = models.inception_v3()

        if is_freeze:
            for params in self.model.parameters():
                params.requires_grad = False
        if model_name == "mobilenetv3":
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features=576,out_features=1024,bias=True),
                nn.Hardswish(),
                nn.Dropout(p = 0.2,inplace=True),
                nn.Linear(in_features=1024,out_features=num_classes,bias=True)
            )
        elif model_name == "resnet18" or model_name == "inceptionv3":
            self.model.fc = nn.Sequential(
                nn.Linear(in_features=512,out_features=num_classes,bias=True)
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0, std = 0.01)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        out = self.model(x)
        out = torch.softmax(out,dim=1)
        return out

if __name__ == '__main__':
    model = modelFineTune(model_name="mobilenetv3",is_freeze=False,pretrained=False)
    from torchinfo import summary
    summary(model,input_size=(4,3,224,224))
    pass