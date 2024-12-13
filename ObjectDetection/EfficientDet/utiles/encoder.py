"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/2/6 21:25
"""

import torch
import torch.nn as nn
from configs.config import DefaultConfig

def coords_fmap2orig(feature,stride):
    """
    feature: 对应level的特征层输出结果
    stride: 对应level相比于原图下采样倍数
    """
    c,h,w,num_classes = feature.size()
    shifts_x = torch.arange(0, w * stride, stride, dtype = torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype = torch.float32)
    shift_y,shift_x = torch.meshgrid(shifts_y,shifts_x)
    shift_x,shift_y = shift_x.contiguous().view(-1),shift_y.contiguous().view(-1)
    #加上stride // 2，将左上角的网格坐标转换到中心点
    coords = torch.stack([shift_x,shift_y],dim = -1) + stride // 2
    return coords

class Encoder(nn.Module):
    __annotations__ = {
        'strides': [[8, 16, 32, 64, 128]],  # P3, P4, P5, P6 and P7
        'limit_range': [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
    }
    def __init__(self,strides,limit_ranges):
        super(Encoder, self).__init__()
        self.strides = strides
        self.limit_ranges = limit_ranges
        assert len(strides) == len(limit_ranges),"strides' size should equal limit_ranges"
    def forward(self,inputs):
        """
         inputs
               [0]list [cls_logits,cnt_logits,reg_preds]
               cls_logits  list contains four [batch_size,class_num,h,w]
               cnt_logits  list contains four [batch_size,1,h,w]
               reg_preds   list contains four [batch_size,4,h,w]
               [1]gt_boxes [batch_size,m,4]  FloatTensor
               [2]classes [batch_size,m]  LongTensor
        Returns
               cls_targets:[batch_size,sum(_h*_w),1]
               cnt_targets:[batch_size,sum(_h*_w),1]
               reg_targets:[batch_size,sum(_h*_w),4]
        """
        cls_logits,center_logits,loc_logits = inputs[0]
        loc_targets,cls_targets = inputs[1],inputs[2]
        cls_target_all_level = []
        loc_target_all_level = []
        center_target_all_level = []
        for level in range(len(cls_logits)):
            level_out = [cls_logits[level],center_logits[level],loc_logits[level]]
            level_targets = self.gen_level_targets(
                level_out,loc_targets,cls_targets,self.strides[level],self.limit_ranges[level]
            )
            cls_target_all_level.append(level_targets[0])
            loc_target_all_level.append(level_targets[1])
            center_target_all_level.append(level_targets[2])
        return torch.cat(cls_target_all_level,dim=1),torch.cat(loc_target_all_level,dim=1),\
               torch.cat(center_target_all_level,dim=1)
    def gen_level_targets(self,outputs,loc_targets,cls_targets,stride,limit_range,sample_radiu_ratio = 1.5):
        """
            out list contains: [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]
            gt_boxes: [batch_size,m,4]
            classes: [batch_size,m]
            stride: int
            limit_range: list [min,max]
            Returns:
                cls_targets,cnt_targets,reg_targets
        """
        cls_logits,center_logits,loc_logits = outputs
        batch_size = cls_logits.shape[0]
        num_classes = cls_logits.shape[1]
        num_boxes = loc_targets.shape[1]
        #[b,num_classes,h,w] => [b,h,w,num_classes]
        cls_logits = cls_logits.permute(0,2,3,1)
        coords = coords_fmap2orig(cls_logits,stride).to(device = cls_targets.device)
        cls_logits = cls_logits.view((batch_size,-1,num_classes))
        center_logits = center_logits.permute(0,2,3,1)
        center_logits = center_logits.view((batch_size,-1,1))
        loc_logits = loc_logits.permute(0,2,3,1).view((batch_size,-1,4))

        h_mul_w = cls_logits.shape[1]
        x = coords[:,0]
        y = coords[:,1]
        #[1,hw,1] - [b,1,m] => [b,hw,m] 计算gt boxes的左上角右下角和坐标相对于划分的中心点偏移
        l_off = x[None,:,None] - loc_targets[...,0][:,None,:]
        t_off = y[None,:,None] - loc_targets[...,1][:,None,:]
        r_off = loc_targets[...,2][:,None,:] - x[None,:,None]
        b_off = loc_targets[...,3][:,None,:] - y[None,:,None]
        #[b,hw,m,4]
        ltrb_off = torch.stack([l_off,t_off,r_off,b_off],dim = -1)
        #[b,hw,m]计算gt boxes的面积
        areas = (ltrb_off[...,0] + ltrb_off[...,2]) * (ltrb_off[...,1] + ltrb_off[...,3])
        #[b,hw,m]
        off_min = torch.min(ltrb_off,dim = -1)[0]
        off_max = torch.max(ltrb_off,dim = -1)[0]
        # 判断当前的边界框的[l*,t*,r*,b*]中最小值和最大值和给定的范围之间是否满足关系
        mask_in_gtboxes = off_min > 0
        mask_in_level = (off_min > limit_range[0])&(off_max <= limit_range[1])

        radius = stride * sample_radiu_ratio
        gt_center_x = (loc_targets[...,0] + loc_targets[...,2]) / 2
        gt_center_y = (loc_targets[...,1] + loc_targets[...,3]) / 2

        #计算gt boxes的中心点偏移
        c_l_off = x[None,:,None] - gt_center_x[:,None,:]
        c_t_off = y[None,:,None] - gt_center_y[:,None,:]
        c_r_off = gt_center_x[:,None,:] - x[None,:,None]
        c_b_off = gt_center_y[:,None,:] - y[None,:,None]
        c_ltrb_off = torch.stack([c_l_off,c_t_off,c_r_off,c_b_off],dim = -1)
        #计算中心点偏移最大值
        c_off_max = torch.max(c_ltrb_off,dim=-1)[0]
        #最大中心点偏移在给定的半径范围之内设置为正样本
        mask_center = c_off_max < radius

        mask_pos = mask_in_gtboxes&mask_in_level&mask_center
        areas[~mask_pos] = 9999999
        #得到最小面积的索引，最小面积的索引作为正样本
        areas_min_ind = torch.min(areas,dim=-1)[1]
        #将areas_min_ind索引位置设置为1
        reg_targets = ltrb_off[
            torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)
        ]
        reg_targets = reg_targets.view(batch_size,-1,4)
        #[b,hw,m]
        """
        broadcast_tensors:
            torch.broadcast_tensors 是 PyTorch 中的一个将tensor扩充的函数
            在a, b = torch.broadcast_tensors(x, y)中，是将x与y的形状“黏合”起来，从而组成两个形状相同的tensor：a、b，
            在不复制数据的情况下就能进行运算，整个过程可以做到避免无用的复制，达到更高效的运算。
        """
        cls_targets = torch.broadcast_tensors(cls_targets[:,None,:],areas.long())[0]
        cls_targets = cls_targets[
            torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)
        ]
        cls_targets = cls_targets.view(batch_size,-1,1)

        left_right_min = torch.min(reg_targets[...,0],reg_targets[...,2])
        left_right_max = torch.max(reg_targets[...,0],reg_targets[...,2])
        top_bottom_min = torch.min(reg_targets[...,1],reg_targets[...,3])
        top_bottom_max = torch.max(reg_targets[...,1],reg_targets[...,3])
        center_targets = ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(dim = -1)

        assert reg_targets.shape == (batch_size,h_mul_w,4)
        assert cls_targets.shape == (batch_size,h_mul_w,1)
        assert center_targets.shape == (batch_size,h_mul_w,1)

        mask_pos_2 = mask_pos.long().sum(dim=-1)
        mask_pos_2 = mask_pos_2 >= 1
        assert  mask_pos_2.shape == (batch_size,h_mul_w)
        cls_targets[~mask_pos_2] = 0
        reg_targets[~mask_pos_2] = -1
        center_targets[~mask_pos_2] = -1
        return cls_targets,center_targets,reg_targets