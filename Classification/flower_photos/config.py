"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/7/29-15:15
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import torch.cuda

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = r'C:\TransferLearning\ClassificationDataset\flower_photos'
OUTPUT = r'./output'
NUM_CLASSES =  10
IMG_SIZE = 224
IN_CHANNELS = 3
LR = 0.0001
EPOCH = 100
START_EPOCH = 0
RESUME = r''
EVAL_EPOCH = 5
BEST_ACC = 0.
BATCH_SIZE = 4

index_map_name = {
    0:'daisy',1:'dandelion',
    2:'roses',3:'sunflowers',4:'tulips'
}

name_map_index = {
    'daisy':0,'dandelion':1,
    'roses':2,'sunflowers':3,'tulips':4
}

name = ['黛西','蒲公英','玫瑰','向日葵','郁金香']
