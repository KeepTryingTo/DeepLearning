import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from nets.common import CenternetDeconv
from utils.centernet import pseudo_nms




def switch_backbones(bone_name):
    from nets.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, \
        resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
    if bone_name == "resnet18":
        return resnet18()
    elif bone_name == "resnet34":
        return resnet34()
    elif bone_name == "resnet50":
        return resnet50()
    elif bone_name == "resnet101":
        return resnet101()
    elif bone_name == "resnet152":
        return resnet152()
    elif bone_name == "resnext50_32x4d":
        return resnext50_32x4d()
    elif bone_name == "resnext101_32x8d":
        return resnext101_32x8d()
    elif bone_name == "wide_resnet50_2":
        return wide_resnet50_2()
    elif bone_name == "wide_resnet101_2":
        return wide_resnet101_2()
    else:
        raise NotImplementedError(bone_name)



class SingleHead(nn.Module):
    def __init__(self, in_channel,
                 inner_channel,
                 out_channel, num_convs,
                 bias_fill=False, bias_value=0):
        super(SingleHead, self).__init__()
        head_convs=[]
        for i in range(num_convs):
            inc = in_channel if i==0 else inner_channel
            head_convs.append(nn.Conv2d(in_channels=inc,
                                        out_channels=inner_channel,
                                        kernel_size=3, padding=1))
            head_convs.append(nn.ReLU())
        head_convs.append(nn.Conv2d(inner_channel, out_channel, kernel_size=1))

        self.head_convs=nn.Sequential(*head_convs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if bias_fill:
            self.head_convs[-1].bias.data.fill_(bias_value)

    def forward(self, x):
        return self.head_convs(x)



class CenternetHead(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    :return
    cls: shape=[bs,num_cls,h,w]
    wh: shape=[bs,4,h,w]  4==>(l,t,r,b)对应原图尺度中心点到四个边的距离
    """
    def __init__(self, num_cls, cls_num_convs, wh_num_convs, bias_value,
                 wh_offset_base=16,topk=100,score_thr=0.01,down_ratio=4):
        super(CenternetHead, self).__init__()
        self.wh_offset_base = wh_offset_base
        self.topk=topk
        self.score_thr=score_thr
        self.down_ratio=down_ratio
        self.cls_head = SingleHead(
            64,
            inner_channel=128,
            out_channel=num_cls,
            num_convs=cls_num_convs,
            bias_fill=True,
            bias_value=bias_value,
        )
        self.wh_head = SingleHead(
            64,
            inner_channel=64,
            out_channel=4,
            num_convs=wh_num_convs,
        )

    def _topk(self, scores, topk):
        batch, cat, height, width = scores.size()

        # TODO 在topk的维度上求解预测分数最大的，并选择前topk个 => both are (batch, 80, topk)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)

        #TODO 获得X和Y轴坐标
        topk_inds = topk_inds % (height * width)
        topk_ys = (1.0 * topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # TODO 在最后的维度上继续求解最大分数 得到最终预测的分数
        #  both are (batch, topk). select topk from 80*topk
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
        #TODO 得到预测的类别
        topk_clses = (1.0 * topk_ind / topk).int()
        topk_ind = topk_ind.unsqueeze(2)
        topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    @torch.no_grad()
    def get_bboxes(self,
                   pred_heatmap,
                   pred_wh,
                   ):
        #TODO 得到预测的heatmap和预测的高宽
        batch, cat, height, width = pred_heatmap.size()
        pred_heatmap = pred_heatmap.detach().sigmoid_()
        wh = pred_wh.detach()

        # TODO 对heatmap进行解码操作 perform nms on heatmaps
        heat = pseudo_nms(pred_heatmap)  # used maxpool to filter the max score

        # (batch, topk)
        scores, inds, clses, ys, xs = self._topk(heat, topk=self.topk)
        #TODO 根据下采样大小将解码得到的坐标还原回相对原图大小
        xs = xs.view(batch, self.topk, 1) * self.down_ratio
        ys = ys.view(batch, self.topk, 1) * self.down_ratio

        wh = wh.permute(0, 2, 3, 1).contiguous()
        wh = wh.view(wh.size(0), -1, wh.size(3))
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), wh.size(2))
        wh = wh.gather(1, inds)

        #TODO 解码得到最终的预测结果
        wh = wh.view(batch, self.topk, 4)
        clses = clses.view(batch, self.topk, 1).float()
        scores = scores.view(batch, self.topk, 1)

        bboxes = torch.cat([xs - wh[..., [0]], ys - wh[..., [1]],
                            xs + wh[..., [2]], ys + wh[..., [3]]], dim=2)

        result_list = []
        #TODO 遍历batch维度
        for batch_i in range(bboxes.shape[0]):
            scores_per_img = scores[batch_i]
            #TODO 进一步过滤掉那些低置信度的框
            scores_keep = (scores_per_img > self.score_thr).squeeze(-1)

            scores_per_img = scores_per_img[scores_keep]
            bboxes_per_img = bboxes[batch_i][scores_keep]
            labels_per_img = clses[batch_i][scores_keep]

            bboxes_per_img = torch.cat([bboxes_per_img,
                                        scores_per_img], dim=1)
            ret=torch.cat([bboxes_per_img,labels_per_img],dim=-1)  #shape=[num_boxes,6] 6==>x1,y1,x2,y2,score,label
            result_list.append(ret)

        return result_list


    def forward(self, x):
        cls = self.cls_head(x)
        wh = F.relu(self.wh_head(x))*self.wh_offset_base
        #TODO 判断是训练阶段还是测试阶段，如果是测试阶段就需要对预测结果解码
        if self.training:
            return cls,wh
        else:
            results = self.get_bboxes(cls,wh)
            return results

class CenterNet(nn.Module):
    """
    Implement CenterNet (https://arxiv.org/abs/1904.07850).
    """
    def __init__(self,
                 num_cls=80,
                 PIXEL_MEAN=[0.485, 0.456, 0.406],
                 PIXEL_STD=[0.229, 0.224, 0.225],
                 backbone='resnet50',
                 cfg=None
                 ):
        super(CenterNet,self).__init__()
        self.cfg = cfg
        self.num_cls = num_cls
        self.mean = PIXEL_MEAN
        self.std = PIXEL_STD
        self.backbone = switch_backbones(backbone)
        c2, c3, c4, c5 = self.backbone.inner_channels
        self.upsample = CenternetDeconv(channels=self.cfg['DECONV_CHANNEL'],
                                        shortcut_in_channels = [c4, c3, c2],
                                        modulate_deform=self.cfg['MODULATE_DEFORM'])
        self.head = CenternetHead(num_cls, self.cfg['cls_num_convs'], self.cfg['wh_num_convs'],self.cfg['BIAS_VALUE'],
                                  topk=self.cfg['max_per_img'],score_thr=self.cfg['score_thr'],
                                  down_ratio=self.cfg['down_ratio'])

    def forward(self, x):
        '''
        note: 作验证或者推理时,x.shape=[1,C,H,W]
        :param x:
        :return:
        '''
        features = self.backbone(x)
        up_fmap = self.upsample(features)
        pred_dict = self.head(up_fmap)

        return pred_dict

def demo():
    input_tensor = torch.randn(size=(1, 3, 512, 512)).cuda()
    centernet_cfg=dict(
        DECONV_CHANNEL=[2048, 256, 128, 64],
        DECONV_KERNEL=[4, 4, 4],
        NUM_CLASSES=80,
        MODULATE_DEFORM=True,
        BIAS_VALUE=-2.19,
        DOWN_SCALE=4,
        MIN_OVERLAP=0.7,
        TENSOR_DIM=128,
        cls_num_convs=2,
        wh_num_convs=2,
        max_per_img=100,
        score_thr=0.01,
        down_ratio=4
    )
    net = CenterNet(backbone="resnet50",cfg=centernet_cfg).cuda()
    net.train()
    cls_out,wh_out = net(input_tensor)
    print(cls_out.shape,cls_out.dtype,cls_out.device)
    print(wh_out.shape,wh_out.dtype,wh_out.device)

if __name__ == '__main__':
    demo()
    pass

    # net.eval()
    # pred_dict=net(input_tensor)
    # boxes,scores,classes=pred_dict['pred_boxes'],pred_dict['scores'],pred_dict['pred_classes']
    # print('pred box info: ',boxes.shape,boxes.dtype,boxes.device)
    # print('score info: ',scores.shape,scores.dtype,scores.device)
    # print('pred classes info: ',classes.shape,classes.dtype,classes.device)
    # print(boxes.max(),boxes.min())
    # print(scores.max(),scores.min())
    # print(classes.max(),classes.min())

    # backbone = switch_backbones('resnet34')
    # for name,param in backbone.named_parameters():
    #     print(name,param.requires_grad)









