import torch
from torch import nn
from configs.defaults import _C as cfg
from models.resnet import resnet101
from models.decoder import DSSDDecoder
from models.box_predictor import BoxPredictor
from models.dssdBoxHead import DSSDBoxHead

def createCfg(config_file = r'configs/resnet101_dssd320_voc0712.yaml'):
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

class DSSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = resnet101(pretrained=True)
        self.decoder = DSSDDecoder(cfg)
        self.box_head = DSSDBoxHead(cfg)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        features = self.decoder(features)
        if self.training:
            detections, loss_dict = self.box_head(features, targets)
            return loss_dict
        else:
            detections, cls_logits, bbox_pred = self.box_head(features)
        return detections,cls_logits, bbox_pred

def demo():
    cfg = createCfg(config_file = r'../configs/resnet101_dssd320_voc0712.yaml')
    model = DSSDDetector(cfg=cfg)
    model.eval()
    x = torch.zeros(size = (2,3,320,320))
    detections,cls_logits, bbox_pred = model(x)

    for detection in detections:
        print('boxes.shape: {}'.format(detection['boxes'].size()))
        print('confidence.shape: {}'.format(detection['scores'].size()))
        print('labels.shape: {}'.format(detection['labels'].size()))

    # from torchinfo import summary
    # summary(model,input_size=(2,3,320,320))

if __name__ == '__main__':
    demo()
    pass
