"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/31-20:06
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os.path

def split_dataset(file_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    datas = []
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            if not line or line == "":
                continue
            datas.append(line)
    train = datas[0:10000]
    val = datas[10000:11000]
    with open(os.path.join(output_path, "train.json"), "w", encoding="utf-8") as w:
        for line in train:
            w.write(line)
            w.flush()

    with open(os.path.join(output_path, "val.json"), "w", encoding="utf-8") as w:
        for line in val:
            w.write(line)
            w.flush()
    print("train count: ", len(train))
    print("val count: ", len(val))


if __name__ == '__main__':
    file_path = "data/train.json"
    split_dataset(file_path=file_path, output_path="data")