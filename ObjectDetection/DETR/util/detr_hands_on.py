"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/1/6 15:26
"""

import os
import time
import torch
import requests
from torch import nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.models import resnet50
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

    # model 仅支持高宽比范围为： 0.5 ～ 2, 如果你想 使用 高宽比超过这个范围的图像，你得重新 rescale 你的图像
    # 并且使得最大边长不超过 1333， 这样才能取得较好的检测效果
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    #[bs,num_queries,num_classes]
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1] #对类别求解最大概率值
    #得到每一个box的类别概率最大值并且过滤掉小于0.7的box
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales pred_boxes: [bs,num_queries,4]
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], probas,keep ,bboxes_scaled ,outputs


def plot_results(pil_img, prob, boxes,img_name):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        #得到最大概率值对应的类别索引
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(f'../runs/{img_name.split(".")[0]}.png')
    plt.show()


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#加载模型方式一
# model = DETRdemo(num_classes=91)
# state_dict = torch.hub.load_state_dict_from_url(
#     url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
#     map_location='cpu',
#     check_hash=True
# )
# model.load_state_dict(state_dict)
# model.eval()

#加载模型方式二
# model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
# model.eval()

#加载模型方式三
from hubconf import detr_resnet50
checkpoint = torch.load(r'../weights/detr-r50-e632da11.pth',map_location='cpu')
model = detr_resnet50(pretrained=False,num_classes=91)
model.load_state_dict(checkpoint['model'])
model.eval()

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

root = r'../images/dog01.png'
img = Image.open(root).convert('RGB')
start_time = time.time()
scores,probas,keep, bboxes_scaled,outputs = detect(img, model, transform)
end_time = time.time()
plot_results(img, scores, bboxes_scaled,'img_name')
print('inference time: {}'.format(end_time - start_time))

"""
这里我们可视化最后一个解码器层的注意权重。这对应于可视化，
对于每个检测到的对象，模型正在查看图像的哪个部分来预测这个特定的边界框和类。
我们将使用钩子从变压器中提取注意力权重(所有头部的平均值)。
"""
#从transformer中提取
# use lists to store the outputs via up-values
conv_features, enc_attn_weights, dec_attn_weights = [], [], [] #分别用于保存卷积层，编码层以及解码层的权重信息

hooks = [
    model.backbone[-2].register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    ),
    model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    ),
    model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output[1])
    ),
]


for hook in hooks:
    hook.remove()

# don't need the list anymore
conv_features = conv_features[0]
enc_attn_weights = enc_attn_weights[0]
dec_attn_weights = dec_attn_weights[0]

#可视化结果
# get the feature map shape
h, w = conv_features['0'].tensors.shape[-2:]

fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
colors = COLORS * 100
for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    ax = ax_i[0]
    ax.imshow(dec_attn_weights[0, idx].view(h, w))
    ax.axis('off')
    ax.set_title(f'query id: {idx.item()}')
    ax = ax_i[1]
    ax.imshow(img)
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, color='blue', linewidth=3))
    ax.axis('off')
    ax.set_title(CLASSES[probas[idx].argmax()])
fig.tight_layout()


"""
在本节中，我们可视化模型的自关注。这使我们能够直观地了解模型如何对对象进行编码。
特别地，我们从注意反应图中看到编码器已经参与到实例分离过程中。
让我们从可视化编码器自关注的形状开始:
"""
# output of the CNN
f_map = conv_features['0']
print("Encoder attention:      ", enc_attn_weights[0].shape)
print("Feature map:            ", f_map.tensors.shape)


"""
我们可以看到，自注意是一个大小为[H * W, H * W]的方阵，
其中H和W是特征映射的大小，因此我们对其进行重塑，使其具有更可解释的表示[H, W, H, W]。
"""
# get the HxW shape of the feature maps of the CNN
shape = f_map.tensors.shape[-2:]
# and reshape the self-attention to a more interpretable shape
sattn = enc_attn_weights[0].reshape(shape + shape)
print("Reshaped self-attention:", sattn.shape)


"""
既然我们已经把自我关注变成了一种更容易解释的表征，
让我们试着把它形象化。考虑到自我注意是一个非常大的矩阵，
让我们从选择几个参考点开始，我们将在这些参考点上观察注意力。
"""
# downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
fact = 32

# let's select 4 reference points for visualization
idxs = [(200, 200), (280, 400), (200, 600), (440, 800),]

# here we create the canvas
fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
# and we add one plot per reference point
gs = fig.add_gridspec(2, 4)
axs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[0, -1]),
    fig.add_subplot(gs[1, -1]),
]

# for each one of the reference points, let's plot the self-attention
# for that point
for idx_o, ax in zip(idxs, axs):
    idx = (idx_o[0] // fact, idx_o[1] // fact)
    ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
    ax.axis('off')
    ax.set_title(f'self-attention{idx_o}')

# and now let's add the central image, with the reference points as red circles
fcenter_ax = fig.add_subplot(gs[:, 1:-1])
fcenter_ax.imshow(img)
for (y, x) in idxs:
    scale = img.height / img.shape[-2]
    x = ((x // fact) + 0.5) * fact
    y = ((y // fact) + 0.5) * fact
    fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
    fcenter_ax.axis('off')
