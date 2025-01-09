from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import tensorflow as tf
import numpy as np
import os
import cv2
import pickle
from codes.miscc.utils import get_image
import scipy.misc
from PIL import Image
import pandas as pd

# from glob import glob

# TODO: 1. current label is temporary, need to change according to real label
#       2. Current, only split the data into train, need to handel train, test

LR_HR_RETIO = 4
IMSIZE = 256
LOAD_SIZE = int(IMSIZE * 76 / 64)
BIRD_DIR = r'/home/ff/myProject/KGT/myProjects/myDataset/text2image/birds'


def load_filenames(data_dir):
    filepath = data_dir + 'filenames.pickle'
    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    return filenames


def load_bbox(data_dir):
    bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
    df_bounding_boxes = pd.read_csv(bbox_path,
                                    delim_whitespace=True,
                                    header=None).astype(int)
    #
    filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
    df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
    filenames = df_filenames[1].tolist()
    print('Total filenames: ', len(filenames), filenames[0])
    #
    filename_bbox = {img_file[:-4]: [] for img_file in filenames}
    numImgs = len(filenames)
    for i in range(0, numImgs):
        # bbox = [x-left, y-top, width, height]
        bbox = df_bounding_boxes.iloc[i][1:].tolist()

        key = filenames[i][:-4]
        filename_bbox[key] = bbox
    #
    return filename_bbox


def save_data_list(inpath, outpath, filenames, filename_bbox):
    hr_images = []
    lr_images = []
    #TODO 将图像向下采样LR_HR_RETIO倍
    lr_size = int(LOAD_SIZE / LR_HR_RETIO)
    cnt = 0
    for key in filenames:
        bbox = filename_bbox[key]
        f_name = '%s/CUB_200_2011/images/%s.jpg' % (inpath, key)
        #TODO 读取图像并对图像进行一定的处理，进行一定的缩放
        img = get_image(f_name, LOAD_SIZE, is_crop=True, bbox=bbox)
        img = img.astype('uint8')
        hr_images.append(img)
        #TODO 将图像利用双线性插值缩放到指定的大小
        lr_img = cv2.resize(img,dsize = (lr_size, lr_size),interpolation=cv2.INTER_CUBIC)
        lr_images.append(lr_img)
        cnt += 1
        if cnt % 100 == 0:
            print('Load %d......' % cnt)
    #
    print('images', len(hr_images), hr_images[0].shape, lr_images[0].shape)
    #TODO 保存对应高分辨率图像和低分辨率图像
    outfile = outpath + str(LOAD_SIZE) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(hr_images, f_out)
        print('save to: ', outfile)
    #
    outfile = outpath + str(lr_size) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(lr_images, f_out)
        print('save to: ', outfile)


def convert_birds_dataset_pickle(inpath):
    # TODO Load dictionary between image filename to its bbox
    filename_bbox = load_bbox(inpath)
    # ## TODO 加载对应图像名称文件 For Train data
    train_dir = os.path.join(inpath, 'train/')
    train_filenames = load_filenames(train_dir)
    save_data_list(inpath=inpath, outpath=train_dir,
                   filenames=train_filenames,
                   filename_bbox=filename_bbox)

    # ## For Test data
    test_dir = os.path.join(inpath, 'test/')
    test_filenames = load_filenames(test_dir)
    save_data_list(inpath, test_dir, test_filenames, filename_bbox)


if __name__ == '__main__':
    # convert_birds_dataset_pickle(BIRD_DIR)
    pass