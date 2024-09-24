"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/8/16-8:36
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import os.path

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = r'./images'
output = './outputs'
crop_size = 800

className = []
with open(file=r'D:\conda3\Transfer_Learning\B Stand\day18\main\Flask_deploy\detection\classes.txt', mode='r', encoding='utf-8') as fp:
    lines = fp.readlines()

for line in lines:
    className.append(line.strip('\n'))

print(len(className))
print(className)