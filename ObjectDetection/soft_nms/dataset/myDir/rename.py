"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023-01-02 11:25
"""

import os
import cv2
import time

def RenameFilePng(filePath,savePath):
    """
    :param filePath:要修改名称的文件名存在的路径
    :param savePath: 保存的文件路径
    :return: 无返回值
    """
    #起始时间
    startTime=time.time()
    #列出给定路径下面的所有文件
    fileList=os.listdir(filePath)
    for i,file in enumerate(fileList):
        # 路径进行拼接
        imgfile = str(i) + '.png'
        # 打开文件
        openimg = os.path.join(filePath,file)
        newImg = cv2.imread(openimg)
        # 以PNG格式保存到指定的路径
        saveImg = os.path.join(savePath, imgfile)
        print(saveImg)
        # saveImg=data/tempFile/i.png...
        cv2.imwrite(saveImg, newImg)
        print('正在转换...')
    print("完成转换!\n")
    #完成结束时间
    endTime=time.time()
    print('完成时间: {}'.format(endTime-startTime))

def RenameFileTxt(filePath,savePath):
    """
    :param filePath:要修改名称的文件名存在的路径
    :param savePath: 保存的文件路径
    :return: 无返回值
    """
    #起始时间
    startTime=time.time()
    #列出给定路径下面的所有文件
    fileList=os.listdir(filePath)
    for i,file in enumerate(fileList):
        # 路径进行拼接
        imgfile = str(i) + '.txt'
        # 打开文件
        openimg = os.path.join(filePath,file)
        writeContext=[]
        with open(openimg,'r') as fp:
            imageStr=fp.readlines()
            for cont in imageStr:
                writeContext.append(cont)
        # 以PNG格式保存到指定的路径
        saveImg = os.path.join(savePath, imgfile)
        print(saveImg)
        with open(saveImg,'a') as fp:
            for boxes in writeContext:
                fp.write(boxes)
    print("完成转换!\n")
    #完成结束时间
    endTime=time.time()
    print('完成时间: {}'.format(endTime-startTime))

if __name__ == '__main__':
    # RenameFileTxt(filePath='data/widerFace/tempFile',savePath=r'data/widerFace/wider_face_train_txt')
    RenameFilePng(filePath='data/temp',savePath='data/resNet50_Classify/train/1910300715')
    # imageList=os.listdir('data/train/person')
    # print(len(imageList))
    pass

