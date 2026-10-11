'''
# -*- encoding: utf-8 -*-
# 文件    : loss.py
# 说明    : 自定义损失函数
# 时间    : 2022/07/01 14:09:48
# 作者    : Hito
# 版本    : 1.0
# 环境    : TensorFlow2.3 or pytorch1.7
'''

import torch




def clip_by_tensor(t,t_min,t_max):
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def BCELoss(pred,target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output