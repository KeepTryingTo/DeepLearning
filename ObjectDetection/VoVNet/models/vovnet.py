"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/11/19-9:54
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict
from models.classification.vovnet import vovnet57

def initStateDict(weight_path = r'../../weights/vovnet57_torchvision.pth'):
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = OrderedDict()
    for key, value in checkpoint.items():
        if 'module.' in key:
            state_dict[key[7:]] = value
        else:
            state_dict[key] = value

    return state_dict

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class DenseSupervision1(nn.Module):

    def __init__(self,inC,outC=256):
        super(DenseSupervision1, self).__init__()
        self.model_name='DenseSupervision'

        self.right = nn.Sequential(
            # nn.BatchNorm2d(inC),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(inC,outC,1),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC,outC,kernel_size=1,bias=False)
        )

    def forward(self,x1,x2):
        # x1 should be f1
        right = self.right(x1)
        return torch.cat([x2,right],1)


class DenseSupervision(nn.Module):

    def __init__(self,inC,outC=128):
        super(DenseSupervision, self).__init__()
        self.model_name='DenseSupervision'
        self.left = nn.Sequential(
            nn.MaxPool2d(2,2,ceil_mode=True),
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC,outC,1,bias=False)
        )
        self.right = nn.Sequential(
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC,outC,1,bias=False),

            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(outC,outC,3,2,1,bias=False)
        )

    def forward(self,x):
        left = self.left(x)
        right = self.right(x)
        return torch.cat([left,right],1)

class VOVNet_Backbone(nn.Module):
    def __init__(self,pretrained_path = r'../../weights/vovnet57_torchvision.pth'):
        super().__init__()
        self.pretrained_path = pretrained_path
        if self.pretrained_path:
            state_dict = initStateDict(self.pretrained_path)
            model = vovnet57(pretrained=False)
            model.load_state_dict(state_dict)
        else:
            model = vovnet57(pretrained=False)

        self.stem_stage1 = model.stem
        self.stage2 = model.stage2
        self.stage3 = model.stage3
        self.stage4 = model.stage4
        self.stage5 = model.stage5

        self.dense_sup1 = DenseSupervision1(768, 1024)
        self.dense_sup2 = DenseSupervision(2048, 512)
        self.dense_sup3 = DenseSupervision(1024, 256)
        self.dense_sup4 = DenseSupervision(512, 128)
        self.dense_sup5 = DenseSupervision(256, 128)

    def forward(self,x):
        out = self.stem_stage1(x)
        out = self.stage2(out)

        out = self.stage3(out)
        f1 = out

        out = self.stage4(out)
        f2 = out

        out = self.stage5(out)

        """
        f1.shape: torch.Size([1, 512, 37, 37])
        f2.shape: torch.Size([1, 768, 18, 18])
        out.shape: torch.Size([1, 1024, 9, 9])
        """
        # print('f1.shape: {}'.format(f1.size()))
        # print('f2.shape: {}'.format(f2.size()))
        # print('out.shape: {}'.format(out.size()))


        f3 = self.dense_sup1(f2, out)
        f4 = self.dense_sup2(f3)
        f5 = self.dense_sup3(f4)
        f6 = self.dense_sup4(f5)
        return f1, f2, f3, f4, f5, f6




class VOVNet(nn.Module):
    def __init__(self,num_classes,
                 num_anchors,
                 pretrained_path = None):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.backbone = VOVNet_Backbone(pretrained_path=pretrained_path)

        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()

        in_channels = [512, 768, 2048, 1024, 512, 256]
        # num_anchors = (4, 6, 6, 6, 4, 4)
        for i, (inC, num_anchor) in enumerate(zip(in_channels, num_anchors)):
            self.loc_layers += [nn.Sequential(nn.Conv2d(inC,
                                                num_anchor * 4, kernel_size=3,
                                                padding=1, bias=False),
                                                nn.BatchNorm2d(num_anchor * 4)
                                              )]
            self.cls_layers += [nn.Sequential(nn.Conv2d(inC,
                                                num_anchor * num_classes,
                                                kernel_size=3, padding=1,
                                                bias=False),
                                                nn.BatchNorm2d(num_anchor * num_classes)
                                              )]
        self.normalize = nn.ModuleList([L2Norm(chan, 20) for chan in in_channels])

    def reset_parameters(self):
        for name,param in self.extractor.named_parameters():
            if hasattr(param,'weight'):
                nn.init.xavier_uniform(param.weight.data,gain=nn.init.calculate_gain('relu'))

        for name,param in self.loc_layers.named_parameters():
            if hasattr(param,'weight'):
                nn.init.normal(param.weight.data,std=0.01)

        for name,param in self.cls_layers.named_parameters():
            if hasattr(param,'weight'):
                nn.init.normal(param.weight.data,std=0.01)


    def forward(self,x):
        loc_preds = []
        cls_preds = []
        xs = self.backbone(x)
        for i, x in enumerate(xs):
            x = self.normalize[i](x)
            loc_pred = self.loc_layers[i](x)

            """
                out.shape: torch.Size([1, 512, 37, 37])
                out.shape: torch.Size([1, 768, 18, 18])
                out.shape: torch.Size([1, 2048, 9, 9])
                out.shape: torch.Size([1, 1024, 5, 5])
                out.shape: torch.Size([1, 512, 3, 3])
                out.shape: torch.Size([1, 256, 2, 2])
            """
            # print('loc_pre.shape: {}'.format(loc_pred.size()))

            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))

            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        return loc_preds, cls_preds


def demoBackbone():
    x = torch.zeros(size=(1,3,300,300))
    model = VOVNet_Backbone(pretrained_path=r'../weights/vovnet57_torchvision.pth')
    outs = model(x)
    """
        out.shape: torch.Size([1, 512, 37, 37])
        out.shape: torch.Size([1, 768, 18, 18])
        out.shape: torch.Size([1, 2048, 9, 9])
        out.shape: torch.Size([1, 1024, 5, 5])
        out.shape: torch.Size([1, 512, 3, 3])
        out.shape: torch.Size([1, 256, 2, 2])
    """
    for out in outs:
        print('out.shape: {}'.format(out.size()))


def demo():
    x = torch.zeros(size=(1,3,300,300))
    model = VOVNet(num_classes=21,num_anchors=(4, 6, 6, 6, 4, 4),
                   pretrained_path=r'../weights/vovnet57_torchvision.pth')
    outs = model(x)
    for out in outs:
        print('out.shape: {}'.format(out.size()))

    from torchinfo import summary
    summary(model,input_size=(1,3,300,300))

if __name__ == '__main__':
    # demoBackbone()
    demo()
    pass