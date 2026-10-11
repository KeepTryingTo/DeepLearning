import torch
import torch.nn as nn 
import torch.cuda.amp as amp


from rtdetr_pytorch_my.src.core import register
import rtdetr_pytorch_my.src.misc.dist as dist


__all__ = ['GradScaler']

GradScaler = register(amp.grad_scaler.GradScaler)
