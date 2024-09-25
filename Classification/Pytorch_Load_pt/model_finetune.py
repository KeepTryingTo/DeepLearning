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
    def __init__(self,num_classes = 5,pretrained = True,
                 freeze_layers = 5,isFreezeBackbone = False,model_name = 'mobilenetv3'):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.freeze_layers = freeze_layers
        self.isFreezeBackbone = isFreezeBackbone
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

        if isFreezeBackbone:
            for params in self.model.parameters():
                params.requires_grad = False
        else:
            if freeze_layers != -1:
                for params in self.model.features[:freeze_layers].parameters():
                    params.requires_grad = False

        if model_name == "mobilenetv3":
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features=576,out_features=1024,bias=True),
                nn.Hardswish(),
                nn.Dropout(p = 0.2,inplace=True),
                nn.Linear(in_features=1024,out_features=num_classes,bias=True),
                nn.Softmax(dim=-1)
            )
        elif model_name == "resnet18" or model_name == "inceptionv3":
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=512,out_features=num_classes,bias=True),
                nn.Softmax(dim=-1)
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0, std = 0.01)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        out = self.model(x)
        return out

if __name__ == '__main__':
    model = modelFineTune(model_name="mobilenetv3",
                          pretrained=True,freeze_layers=10,isFreezeBackbone=False)
    from torchinfo import summary
    summary(model,input_size=(4,3,224,224))

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'Trainable layer: {name}')
    pass