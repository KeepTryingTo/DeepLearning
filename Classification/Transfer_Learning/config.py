"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/7/29-15:15
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import torch

ROOT = r'D:\conda3\Transfer_Learning\ClassificationDataset\flower_photos'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT = r'./output'
IMG_SIZE = 224
NUM_CLASSES =  5
IN_CHANNELS = 3
BATCH_SIZE = 4
LR = 0.0001
EPOCH = 50
START_EPOCH = 0
RESUME = r''
EVAL_EPOCH = 5
BEST_ACC = 0.
FREEZE_LAYERS = -1
PRETRAINED = False
ISFREEZEBACKBONE = False
index_map_name = {
    0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'
}

name = ['黛西','蒲公英','玫瑰','向日葵','郁金香']