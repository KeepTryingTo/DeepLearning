"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/22-13:51
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import psutil
import torch
from torch import nn

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


def mem_usage():
    """
    Compute the memory usage for the current machine (GB).
    """
    gb = 1 << 30
    mem = psutil.virtual_memory()
    return mem.used / gb