import os
import cv2
import numpy as np
from PIL import Image

def ToBGR(save_path,img_path="data/lfw"):
    #列出指定路径下的所有文件
    imgPdir=os.listdir(img_path).sort(key=lambda x:int(x.split('.')[0]))
    i=15265
    for imgdir in imgPdir:
        #列出当前子文件下的图片(注意这里的路径需要完整的路径，不然找不到图片的路径)
        imgdir=img_path+"\\"+imgdir
        img=os.listdir(imgdir)
        #将当前列出的图片进行转换
        for cimg in img:
            #首先将文件名和文件后准名进行切分
            filename,ext=os.path.splitext(cimg)
            #路径进行拼接
            imgfile=str(i)+'.png'
            i+=1
            #打开文件
            openimg=imgdir+"\\"+cimg
            newImg=cv2.imread(openimg)
            newImg=cv2.resize(newImg,(300,300))
            #以PNG格式保存到指定的路径
            saveImg=os.path.join(save_path,imgfile)
            #saveImg=data/train/001.png...
            cv2.imwrite(saveImg,newImg)
        print('正在转换...')
    print("完成转换!\n")

def ToBGR_1(save_path):
    #列出指定路径下的所有文件
    i=15265
    img=os.listdir('data/tempFile_img')
    img.sort(key=lambda x:int(x.split('.')[0]))
    #将当前列出的图片进行转换
    for cimg in img:
        #首先将文件名和文件后准名进行切分
        filename,ext=os.path.splitext(cimg)
        #路径进行拼接
        imgfile=str(i)+'.png'
        i+=1
        #打开文件
        openimg='data/tempFile_img'+"\\"+cimg
        newImg=cv2.imread(openimg)
        newImg=cv2.resize(newImg,(320,320))
        #以PNG格式保存到指定的路径
        saveImg=os.path.join(save_path,imgfile)
        #saveImg=data/train/001.png...
        cv2.imwrite(saveImg,newImg)
        print('正在转换...')


if __name__ == '__main__':
    # ToBGR_1(save_path="data/ssdLite320_mobilenet_v3_large/train_img")
    pass