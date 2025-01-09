"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/23-23:03
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import pickle

# 打开pickle文件，注意这里使用'rb'模式，表示以二进制读模式打开文件
with open('/home/ff/myProject/KGT/myProjects/myDataset/text2image/birds/captions_DAMSM.pickle', 'rb') as file:
    # 使用pickle.load()函数读取对象
    data = pickle.load(file)

# 现在你可以使用data变量了，它包含了从文件中读取的对象
print(data)

"""
birds/CUB_200_2011
————CU_200_2011/
    ├── images/
    │   ├── 001_class/
    │   │   ├── image_001.jpg
    │   │   ├── image_002.jpg
    │   ├── 002_class/
    │   │   ├── image_001.jpg
    │   │   ├── image_002.jpg
    ├── images.txt(包含每张图像的相对路径)
    |—— classes.txt(包含了每一个类别名称)
    ├── image_class_labels.txt(包含每张图像所对应的类别标签)
    ├── train_test_split.txt(1-表示训练集，0-表示测试集)
    ├── bounding_boxes.txt (包含每张图像物体的坐标框)
    ├── parts/
    │   ├── part_locs.txt (可选)
    |   |—— part_click_locs.txt
    |   |—— parts.txt
    ├── attributes/
    │   ├── attributes.txt (包含图像中物体的属性，比如颜色)

————birds/
    |——CUB_200_2011
    |——DMASMencoder/
    |  |——image_encoder200.pth(对应图像编码器)
    |  |——text_encoder200.pth(图像对应文本内容描述的编码器)
    |——npz/
    |  |——bird_val256_FIDK0.npz(用于模型生成的图像FID的评估)
    |——text/
    |   |── 001_class/
    |   │   │   ├── image_001.txt(图像对应的文本描述句子)
    |   │   │   ├── image_002.txt
    |   │   ├── 002_class/
    |   │   │   ├── image_001.txt
    |   │   │   ├── image_002.txt
    |——test/
    |  |——class_info.pickle(对应测试集图像的类别信息)
    |  |——filenames.pickle(对应测试集图像的路径-不包含后缀名)
    |——train/
    |  |——class_info.pickle()
    |  |——filenames.pickle
    |——captions_DAMSM.pickle(图像对应句子的每个词在词典中的索引)
    
    
————coco/
    |——DMASMencoder/
    |  |——image_encoder100.pth(对应图像编码器)
    |  |——text_encoder100.pth(图像对应文本内容描述的编码器)
    |——npz/
    |  |——coco_val256_FIDK0.npz(用于模型生成的图像FID的评估)
    |——text/
    |   |── 001_class/
    |   │   │   ├── COCO_train2014_000000000009.txt(图像对应的文本描述句子)
    |   
    |——test/
    |  |——filenames.pickle(对应测试集图像的路径-不包含后缀名)
    |——train/
    |  |——filenames.pickle
    |——captions_DAMSM.pickle(图像对应句子的每个词在词典中的索引)    

"""