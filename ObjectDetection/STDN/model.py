import torch
import torch.nn as nn
from layers.scale_transfer_module import ScaleTransferModule
from layers.multibox import MultiBox
from torchvision.models import densenet169
from layers.detection import Detect


"""
different configurations of STDN
"""
stdn_in = {
    '300': [800, 960, 1120, 1280, 1440, 1664],
    '513': [800, 960, 1120, 1280, 1440, 1600, 1664]
}

stdn_out = {
    '300': [(1, 800), (3, 960), (5, 1120), (9, 1280), (18, 360), (36, 104)],
    '513': [(1, 800), (2, 960), (4, 1120), (8, 1280), (16, 1440), (32, 400),
            (64, 104)]
}


class STDN(nn.Module):

    """STDN Architecture"""

    def __init__(self,
                 mode,
                 stdn_config,
                 channels,
                 class_count,
                 anchors,
                 num_anchors,
                 new_size):
        super(STDN, self).__init__()
        self.mode = mode
        #TODO 对应DenseNet169的多层输出的通道数
        self.stdn_in = stdn_in[stdn_config]
        #TODO 经过STM模块之后的特征图大小以及通道数
        self.stdn_out = stdn_out[stdn_config]
        self.channels = channels
        self.class_count = class_count
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.new_size = new_size

        # self.init_weights()

        self.densenet = densenet169(pretrained=True)
        self.scale_transfer_module = ScaleTransferModule(self.new_size)
        self.multibox = MultiBox(num_channels=self.stdn_out,
                                 num_anchors=self.num_anchors,
                                 class_count=self.class_count)

        if mode == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(class_count, top_k=200,
                                 conf_thresh=0.01,
                                 nms_thresh=0.45)

    def get_out_map_sizes(self):
        return [x for x, _ in self.stdn_out]

    def init_weights(self, modules):
        """
        initializes weights for each layer
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        feed forward
        """
        y = self.densenet.features(x)

        output = []
        for stop in self.stdn_in:
            output.append(y[:, :stop, :, :])

        y = self.scale_transfer_module(output)
        class_preds, loc_preds = self.multibox(y)

        if self.mode == 'test':
            output = self.detect.forward(
                self.softmax(class_preds),
                loc_preds,
                self.anchors
            )
        else:
            output = (
                class_preds,
                loc_preds
            )
        return output


def demo():
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    from layers.anchor_box import AnchorBox
    anchor_boxes = AnchorBox(map_sizes=[1, 3, 5, 9, 18, 36],
                             aspect_ratios=[1.6, 2, 3])
    anchor_boxes = anchor_boxes.get_boxes()
    anchor_boxes = anchor_boxes.to(device)

    x = torch.zeros(size=(1,3,300,300))
    model = STDN(
        mode='test',
        stdn_config='300',
        channels=3,
        class_count=21,
        anchors=anchor_boxes,
        num_anchors=8,
        new_size=300
    )

    outs = model(x)
    for out in outs:
        print('out.shape: {}'.format(out.size()))

if __name__ == '__main__':
    demo()
    pass