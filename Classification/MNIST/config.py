"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/7/29-15:15
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT = r'./output'
NUM_CLASSES =  10
IN_CHANNELS = 1
LR = 0.0001
EPOCH = 1
START_EPOCH = 0
RESUME = r''
EVAL_EPOCH = 5
BEST_ACC = 0.
