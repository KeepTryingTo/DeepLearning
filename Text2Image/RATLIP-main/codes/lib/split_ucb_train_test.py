"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/6/14-16:42
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import torch
import numpy as np

root_dir = r'D:\conda3\Transfer_Learning\GANDatasets\birds\CUB_200_2011'
split_train_test_path = r'D:\conda3\Transfer_Learning\GANDatasets\birds\CUB_200_2011\train_test_split.txt'
save_path = r'D:\conda3\Transfer_Learning\GANDatasets\birds\test\filenames.pickle'



train_test_list = []
with open(split_train_test_path,'r',encoding='utf-8') as fp:
    train_test_cls = fp.readlines()

train_nums = 0
test_num = 0
for cls in train_test_cls:
    if int(cls.split(' ')[1]) == 0:
        train_nums += 1
    else:
        test_num += 1

print('train size: {}'.format(train_nums))
print('test size: {}'.format(test_num))

image_path = r'D:\conda3\Transfer_Learning\GANDatasets\birds\CUB_200_2011\images.txt'
list_paths = []
with open(image_path,'r',encoding='utf-8') as fp:
    line_paths = fp.readlines()
for paths in line_paths:
    relative_dir = paths.split(' ')[1].replace('.jpg','')
    list_paths.append(relative_dir)

print('Total size : {}'.format(len(list_paths)))
if __name__ == '__main__':
    pass