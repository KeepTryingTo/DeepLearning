"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023-01-03 7:26
"""

import os
import cv2

#将一张图片为640x480=>224x224并且其中对图像中的坐标进行变换
def equalScalePosition(img_path, txt_path):
    """
    :param img_path: 图片文件的路径
    :param txt_path: 坐标文件的路径
    :return:
    """
    img = cv2.imread(img_path)
    # 获取图片的高宽
    h, w, _ = img.shape
    #计算原图与变换后的图（224x224)进行高宽和比例计算
    ratio_w=float(320/w*1.0)
    ratio_h=float(320/h*1.0)

    # print(ratio_w)
    # print(ratio_h)

    # 读取TXT文件 中的中心坐标和框大小
    with open(txt_path, "r") as fp:
        # 以空格划分
        contline = fp.readline().split(' ')
        # contline : class  x_center y_center width height
        # print(contline)
        # 计算框的左上角坐标和右下角坐标,使用strip将首尾空格去掉
    xmin = float((contline[1]).strip())
    xmax = float(contline[2].strip())
    ymin = float(contline[3].strip())
    ymax = float(contline[4].strip())

            # print((xmin,ymin))
            # print((xmax,ymax))

    #缩放图片之后的坐标变换结果
    xmin_ratio=(ratio_w*xmin)
    xmax_ratio=(ratio_w*xmax)
    ymin_ratio=(ratio_h*ymin)
    ymax_ratio=(ratio_h*ymax)
    # with open(os.path.join('data/widerFace/tempFile_1',txt_), 'a') as fp:
    #     boxes=str(int(xmin_ratio))+' '+str(int(ymin_ratio))+' '+str(int(xmax_ratio))+' '+str(int(ymax_ratio))
    #     fp.write(boxes+'\n')

    # # print((xmin_ratio,ymin_ratio))
    # # print((xmax_ratio,ymax_ratio))
    #
    # # 返回：类别,xleft,yleft,xright,yright
    # return (xmin_ratio,xmax_ratio,ymin_ratio,ymax_ratio)
    return (contline[0],xmin_ratio,xmax_ratio,ymin_ratio,ymax_ratio)

#将所有转换之后的实际坐标保存到一个.txt文件当中
def writeXYToTxt():
    """
    :return:
    """
    imgsPath='data/tempFile_img'
    txtsPath='data/tempFile_txt_300'
    imgs=os.listdir(imgsPath)
    imgs.sort(key=lambda x:int(x.split('.')[0]))
    txts=os.listdir(txtsPath)
    txts.sort(key=lambda x:int(x.split('.')[0]))
    i=15265
    for img_,txt_ in zip(imgs,txts):
        img_path=os.path.join(imgsPath,img_)
        txt_path=os.path.join(txtsPath,txt_)
        tupelist=equalScalePosition(img_path=img_path,txt_path=txt_path)
        datasets = tupelist[0] + ' ' + str(tupelist[1]) + ' ' + str(tupelist[2]) + ' ' + str(tupelist[3]) + ' ' + str(
            tupelist[4])
        with open(os.path.join('data/ssdLite320_mobilenet_v3_large/train_txt',str(i)+'.txt'), 'w') as fp:
            fp.write(datasets)
        i+=1
        print('正在转化...')

        # if tupelist[0]==str(0):
        #     datasets=str(1)+' '+str(tupelist[1])+' '+str(tupelist[2])+' '+str(tupelist[3])+' '+str(tupelist[4])
        #     with open(filename, 'w') as fp:
        #         fp.write(datasets)
        # elif tupelist[0]==str(1):
        #     datasets = str(2) + ' ' + str(tupelist[1]) + ' ' + str(tupelist[2]) + ' ' + str(tupelist[3]) + ' ' + str(
        #         tupelist[4])
        #     with open(filename, 'w') as fp:
        #         fp.write(datasets)
        # elif tupelist[0]==str(2):
        #     datasets = str(3) + ' ' + str(tupelist[1]) + ' ' + str(tupelist[2]) + ' ' + str(tupelist[3]) + ' ' + str(
        #         tupelist[4])
        #     with open(filename, 'w') as fp:
        #         fp.write(datasets)


def draw(img_path,tupelist):
    img = cv2.imread(img_path)
    # 将图片从(640,480)=>(224,224)
    newImg = cv2.resize(img, (224, 224))

    xmin=tupelist[1]
    xmax=tupelist[2]
    ymin=tupelist[3]
    ymax=tupelist[4]
    cv2.rectangle(newImg,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,0,255),1)

    cv2.imshow('newimg',newImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def equalScaleImage(imgPath,savePath):
    """
    :param imgPath: 需要转换图片的位置
    :param savePath: 需要保存转换后图片的位置
    :return:
    """
    imgs=os.listdir(imgPath)
    i=0
    for i,imgName in enumerate(imgs):
        img_path=os.path.join(imgPath,imgName)
        img=cv2.imread(img_path)
        newImg=cv2.resize(img,(224,224))
        save_path=os.path.join(savePath,str(i)+'.png')
        i+=1
        cv2.imwrite(save_path,newImg)
        print('正在转换...')

if __name__ == '__main__':
    # imgsPath='data/train/trainDataset'
    # txtPath='data/XML/trainDataset'
    # imgs=os.listdir(imgsPath)
    # imgs.sort(key=lambda x:int(x.split('.')[0]))
    # txts=os.listdir(txtPath)
    # txts.sort(key=lambda x:int(x.split('.')[0]))
    #
    # for img_,txt_ in zip(imgs,txts):
    #     img_path=os.path.join(imgsPath,img_)
    #     txt_path=os.path.join(txtPath,txt_)
    #     tupeList=equalScalePosition(img_path=img_path,txt_path=txt_path)
    #     draw(img_path=img_path,tupelist=tupeList)
    # writeXYToTxt()
    equalScaleImage(imgPath='data/resNet50_Classify/test',savePath='data/resNet50_Classify')
    pass
