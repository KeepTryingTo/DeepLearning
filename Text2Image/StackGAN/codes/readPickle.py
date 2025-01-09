"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/25-15:10
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import pickle

# 假设你的 .pickle 文件名为 'data.pickle'
filename = '/home/ff/myProject/KGT/myProjects/myDataset/text2image/coco/train/filenames.pickle'

# 打开文件并读取对象
with open(filename, 'rb') as file:
    data = pickle.load(file)

# 现在你可以使用 data 变量了，它包含了从 .pickle 文件中加载的对象
print(data)