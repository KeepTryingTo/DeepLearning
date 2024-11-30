import cv2
import os
from time import sleep
import numpy as np


def Xmin_Xmax_Ymin_Ymax(img_path,txt_path):
    """
    :param img_path: 图片文件的路径
    :param txt_path: 坐标文件的路径
    :return:
    """
    img = cv2.imread(img_path)
    # 获取图片的高宽
    h, w, _ = img.shape
    #读取TXT文件 中的中心坐标和框大小
    with open(txt_path,"r") as fp:
        #以空格划分
        contline=fp.readline().split(' ')
        #contline : class  x_center y_center width height
        # print(contline)
    #计算框的左上角坐标和右下角坐标,使用strip将首尾空格去掉
    xmin=float((contline[1]).strip())-float(contline[3].strip())/2
    xmax=float(contline[1].strip())+float(contline[3].strip())/2

    ymin = float(contline[2].strip()) - float(contline[4].strip()) / 2
    ymax = float(contline[2].strip()) + float(contline[4].strip()) / 2

    #将坐标（0-1之间的值）还原回在图片中实际的坐标位置
    xmin,xmax=w*xmin,w*xmax
    ymin,ymax=h*ymin,h*ymax

    #返回：类别,xleft,yleft,xright,yright
    x1 = float(contline[1])
    x2 = float(contline[2])
    x3 = float(contline[3])
    x4 = float(contline[4])
    # return (contline[0],x1,x2,x3,x4)
    return (contline[0],xmin,xmax,ymin,ymax)

def draw(tupelist):
    img_path = "data/6.png"
    img = cv2.imread(img_path)

    xmin=tupelist[1]
    xmax=tupelist[2]
    ymin=tupelist[3]
    ymax=tupelist[4]
    cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,0,255),2)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#将所有转换之后的实际坐标保存到一个.txt文件当中
def writeXYToTxt():
    """
    :return:
    """
    imgsPath='data/tempFile_img'
    txtsPath='data/tempFile_txt'
    imgs=os.listdir(imgsPath)
    imgs.sort(key=lambda x:int(x.split('.')[0]))
    txts=os.listdir(txtsPath)
    txts.sort(key=lambda x:int(x.split('.')[0]))
    for img_,txt_ in zip(imgs,txts):
        img_path=os.path.join(imgsPath,img_)
        txt_path=os.path.join(txtsPath,txt_)
        tupelist=Xmin_Xmax_Ymin_Ymax(img_path=img_path,txt_path=txt_path)
        datasets=str(1)+' '+str(tupelist[1])+' '+str(tupelist[2])+' '+str(tupelist[3])+' '+str(tupelist[4])
        filename=os.path.join('data/tempFile_txt_1',txt_)
        with open(filename,'w') as fp:
            fp.write(datasets)

#将所有转换之后的实际坐标保存到一个.txt文件当中
def writeXYToTxt_():
    """
    :return:
    """
    imgsPath='data/train/tempFile'
    imgs=os.listdir(imgsPath)
    imgs.sort(key=lambda x: int(x.split('.')[0]))
    i=2206
    for img_ in imgs:
        # 首先将文件名和文件后准名进行切分
        datasets=str(3)+' '+str(0.000)+' '+str(0.000)+' '+str(0.000)+' '+str(0.000)
        filename=os.path.join('data/XML/tempFile', str(i)+'.txt')
        print('写入...')
        i+=1
        with open(filename,'w') as fp:
            fp.write(datasets)

cnt=0
count=0
writeContext = []
fisrt_fileName=''
last_fileName=''
def wider_face_train_bbx_gt():
    global  cnt
    global count
    global writeContext
    global fisrt_fileName
    global last_fileName
    txtPath='data/widerFace/wider_face_split/wider_face_train_bbx_gt.txt'
    savePath='data/widerFace/tempFile'
    with open(txtPath,'r') as fp:
        contlines=fp.readlines()
        for cont in contlines:
            filePath=cont.split('/')
            if len(filePath)==2:
                fisrt_fileName=filePath[1].split('.')[0]
                count+=1
            elif len(cont.split(' '))==11:
                bcont=cont.split(' ')
                if bcont[7]==str(0) and bcont[6]==str(0):
                    xright=int(bcont[0])+int(bcont[2])
                    yright=int(bcont[1])+int(bcont[3])
                    boxes=bcont[0]+' '+bcont[1]+' '+str(xright)+' '+str(yright)
                    writeContext.append(boxes)
            elif len(cont.split(' ')) == 1:
                # print(writeContext)
                file_path=os.path.join(savePath,last_fileName+'.txt')
                if len(writeContext) is not 0:
                    with open(file_path,'a') as f:
                        for boxes in writeContext:
                            f.write(boxes+'\n')
                # print(len(writeContext))
                print('正在写入...')
                last_fileName = fisrt_fileName
                writeContext=[]

if __name__ == '__main__':
   writeXYToTxt()
   # txtPath='data/widerFace/test.txt'
   # with open(txtPath,'r') as fp:
   #     cont=fp.readlines()
   #     for i in cont:
   #         print(i.split(' '))
   #         print(len(i.split(' ')))
   # wider_face_train_bbx_gt()
   pass
   # tupelist=Xmin_Xmax_Ymin_Ymax(img_path='data/6.png',txt_path='data/6.txt')
   # draw(tupelist)
   # writeXYToTxt_()