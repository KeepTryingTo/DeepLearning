import os
import time

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

from PIL import Image
import matplotlib.pyplot as plt
torch.set_grad_enabled(False)

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


class DETRdemo(nn.Module):
    """
    1、这是DETR 的简版实现
    2、与 paper 中的实现相比，有以下几处不同：
        （1）这里使用了 learned positional encoding， 而不是 sine 版本的
        （2）positional encoding is passed at input (instead of attention)
        （3）使用 fc 作为 bbox 的预测器（predictor）， 而不是使用 MLP

    3、这个模型的性能为： （仅支持 batch size = 1）
        （1）在 MSCOCO 验证集上（5k) :  ~40 AP
        （2）使用 Tesla V100 ： runs at ~28 FPS
    """

    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone 加载backbone
        self.backbone = resnet50()
        #去掉最后的顶层部分
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer 加载transformer结构
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        #TODO  output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # TODO spatial positional encodings
        # TODO note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h).sigmoid()}


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def detect(im, model, transform):
    # mean-std normalize the input image (batch size=1)
    img = transform(im).unsqueeze(0)

    # TODO model 仅支持高宽比范围为： 0.5 ～ 2, 如果你想 使用 高宽比超过这个范围的图像，你得重新 rescale 你的图像
    #  并且使得最大边长不超过 1333， 这样才能取得较好的检测效果
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    #[bs,num_queries,num_classes]
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1] #对类别求解最大概率值
    #TODO 得到每一个box的类别概率最大值并且过滤掉小于0.7的box
    keep = probas.max(-1).values > 0.2

    # convert boxes from [0; 1] to image scales pred_boxes: [bs,num_queries,4]
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled


def plot_results(pil_img, prob, boxes,img_name):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    print('boxes.shape: {}'.format(np.shape(boxes)))
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        #得到最大概率值对应的类别索引
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(f'runs/{img_name.split(".")[0]}.png')
    # plt.show()

def build_model():
    def get_args_parser():
        import argparse
        parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--lr_backbone', default=1e-5, type=float)
        # 对于dert lite版本，不支持batch size > 1
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('--epochs', default=300, type=int)
        parser.add_argument('--lr_drop', default=200, type=int)
        parser.add_argument('--clip_max_norm', default=0.1, type=float,
                            help='gradient clipping max norm')

        # Model parameters 给定预训练权重的路径，并且对模型中的mask进行训练
        parser.add_argument('--frozen_weights', type=str, default=None,
                            help="Path to the pretrained model. If set, only the mask head will be trained")
        # * Backbone
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")

        # * Transformer
        parser.add_argument('--enc_layers', default=6, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=6, type=int,
                            help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=2048, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--dropout', default=0.1, type=float,
                            help="Dropout applied in the transformer")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--num_queries', default=100, type=int,
                            help="Number of query slots")
        parser.add_argument('--pre_norm', action='store_true')

        # * Segmentation
        parser.add_argument('--masks', action='store_true',
                            help="Train segmentation head if the flag is provided")

        # Loss
        parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                            help="Disables auxiliary decoding losses (loss at each layer)")
        # * Matcher
        parser.add_argument('--set_cost_class', default=1, type=float,
                            help="Class coefficient in the matching cost")
        parser.add_argument('--set_cost_bbox', default=5, type=float,
                            help="L1 box coefficient in the matching cost")
        parser.add_argument('--set_cost_giou', default=2, type=float,
                            help="giou box coefficient in the matching cost")
        # * Loss coefficients
        parser.add_argument('--mask_loss_coef', default=1, type=float)
        parser.add_argument('--dice_loss_coef', default=1, type=float)
        parser.add_argument('--bbox_loss_coef', default=5, type=float)
        parser.add_argument('--giou_loss_coef', default=2, type=float)
        # 无对象类的相对分类权值
        parser.add_argument('--eos_coef', default=0.1, type=float,
                            help="Relative classification weight of the no-object class")

        # dataset parameters
        parser.add_argument('--dataset_file', default='coco')
        parser.add_argument('--coco_path',
                            default=r'E:\conda_3\PyCharm\Transer_Learning\MSCOCO\coco',
                            type=str)
        parser.add_argument('--coco_panoptic_path', type=str)
        parser.add_argument('--remove_difficult', action='store_true')

        parser.add_argument('--output_dir', default='save_weights',
                            help='path where to save, empty for no saving')
        parser.add_argument('--device', default='cpu',
                            help='device to use for training / testing')
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--resume', default='', help='resume from checkpoint')
        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='start epoch')
        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--num_workers', default=2, type=int)

        # distributed training parameters
        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
        return parser
    args = get_args_parser().parse_args()
    from models.detr import build
    model,_,_ = build(args)
    return model

if __name__ == '__main__':

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    #加载方式一
    # detr = DETRdemo(num_classes=91)
    # state_dict = torch.hub.load_state_dict_from_url(
    #     url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    #     map_location='cpu',
    #     check_hash=True
    # )
    # model.load_state_dict(state_dict)
    # model.eval()

    #加载方式二
    # from hubconf import detr_resnet50
    # state_dict = torch.load(f'weights/detr-r50-e632da11.pth',map_location='cpu')
    # model = detr_resnet50(pretrained=False,num_classes=91,return_postprocessor=False)
    # model.load_state_dict(state_dict['model'])
    # model.eval()

    #加载方式三
    model = build_model()
    state_dict = torch.load(f'/home/ff/myProject/KGT/myProjects/myProjects/DETR/save_weights/checkpoint.pth', map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.eval()

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    root = r'./images'
    img_list = os.listdir(root)
    for img_name in img_list:
        imgPath = os.path.join(root,img_name)
        im = Image.open(imgPath).convert('RGB')
        start_time = time.time()
        scores, boxes = detect(im, model, transform)
        end_time = time.time()
        plot_results(im, scores, boxes,img_name)
        print('inference time: {}'.format(end_time - start_time))



