"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023-01-02 11:12
"""

import os
import cv2
import time

# imgPerson_path='data/tempFile_img'
# txt_path='data/tempFile_txt'
#
# imgPersons=os.listdir(imgPerson_path)
# print(len(imgPersons))
#
# txtPath=os.listdir(txt_path)
# print(len(txtPath))
#
# txts=[]
#
# for txt_path in txtPath:
#     filename,pt=os.path.splitext(txt_path)
#     txts.append(filename)
# print(len(txts))
#
# for img_path in imgPersons:
#     imgName,pt=os.path.splitext(img_path)
#     if imgName not in txts:
#         os.remove(os.path.join(imgPerson_path,img_path))
#         print(imgName)

def Xmin_Xmax_Ymin_Ymax(txt_path):
    """
    :param img_path: 图片文件的路径
    :param txt_path: 坐标文件的路径
    :return:
    """
    img_path = "data/ssd300_vgg16/train_img/15425.png"
    img = cv2.imread(img_path)
    # 获取图片的高宽
    h, w, _ = img.shape
    print(w,h)
    #读取TXT文件 中的中心坐标和框大小
    with open(txt_path,"r") as fp:
        #以空格划分
        contlines=fp.readline().split(' ')
        #返回：类别,xleft,yleft,xright,yright
    x1 = float(contlines[1])
    x2 = float(contlines[2])
    x3 = float(contlines[3])
    x4 = float(contlines[4])
    cv2.rectangle(img, (int(x1), int(x3)), (int(x2), int(x4)), (255, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return (contline[0],xmin,xmax,ymin,ymax)



if __name__ == '__main__':
    Xmin_Xmax_Ymin_Ymax(txt_path='data/ssd300_vgg16/train_txt/15425.txt')
    pass



