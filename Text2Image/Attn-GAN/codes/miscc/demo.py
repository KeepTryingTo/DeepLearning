"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/6/18-13:05
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import pickle

data_dir = r'D:\conda3\Transfer_Learning\GANDatasets\birds\test'
with open(data_dir + '/class_info.pickle', 'rb') as f:
    class_id = pickle.load(f)

print(class_id)