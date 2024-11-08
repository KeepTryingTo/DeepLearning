
import torch
from torch import nn
from models.prediction_module import PredictionModule_C

class BoxPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        #定义记录检测head的列表
        self.prediction_headers = nn.ModuleList()
        #BOXES_PER_LOCATION： [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location
        #OUT_CHANNELS： [1024, 512, 512, 512, 512, 512] # 1, 3, 5, 10, 20, 40
        for level, (boxes_per_location, out_channels) in enumerate(
                zip(cfg.MODEL.PRIORS.BOXES_PER_LOCATION,
                    cfg.MODEL.DECODER.OUT_CHANNELS[::-1])
        ):
            self.prediction_headers.append(
                self.prediction_module(level, out_channels, boxes_per_location)
            )
        self.reset_parameters()

    def prediction_module(self, level, out_channels, boxes_per_location):
        return PredictionModule_C(out_channels, boxes_per_location,
                                  self.cfg.MODEL.NUM_CLASSES)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        cls_logits = []
        bbox_pred = []
        for feature, prediction_header in zip(features, self.prediction_headers):
            _cls, _loc = prediction_header(feature)
            cls_logits.append(_cls.permute(0, 2, 3, 1).contiguous())
            bbox_pred.append(_loc.permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, self.cfg.MODEL.NUM_CLASSES)
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 4)

        return cls_logits, bbox_pred
