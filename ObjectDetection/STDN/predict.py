"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/10/25-9:25
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import os
import time
import cv2
# import cvzone
import numpy as np
import torch
from model import STDN
from layers.anchor_box import AnchorBox
from data.augmentations import BaseTransform
from data.pascal_voc import VOC_CLASSES

conf_thresh = 0.01
root = r'./images'
# save = r'D:\conda3\Transfer_Learning\ObjectDetect\ASSD-Pytorch-master\outputs'
save = r'./outputs'
weight_path = r'./models/2024-11-14 11_43_58.358238/164000.pth'

#TODO 加载预训练模型用于验证
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

anchor_boxes = AnchorBox(map_sizes=[1, 3, 5, 9, 18, 36],
                         aspect_ratios=[1.6, 2, 3])
anchor_boxes = anchor_boxes.get_boxes()
anchor_boxes = anchor_boxes.to(device)

x = torch.zeros(size=(1,3,300,300))
eval_model = STDN(
    mode='test',
    stdn_config='300',
    channels=3,
    class_count=21,
    anchors=anchor_boxes,
    num_anchors=8,
    new_size=300
)
eval_model.load_state_dict(torch.load(weight_path, map_location='cpu'))
eval_model.eval()
eval_model.to(device)
print('Finished loading model!')


transform = BaseTransform(
    size=300,
    mean=(104, 117, 123)
)


def predictImage(conf_thresh = 0.15):
    img_list = os.listdir(root)

    for img_name in img_list:
        start_time = time.time()
        # img = Image.open(os.path.join(root, img_name))
        # w, h = img.size
        # image = transform(
        #     cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # )[0]
        image = cv2.imread(os.path.join(root,img_name))
        h, w, _ = image.shape
        img = image.astype(np.float32)
        # img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img = transform(img)[0]
        img = img[:, :, (2, 1, 0)] #TODO 转换为RGB
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        x = x.to(device)

        with torch.no_grad():
            detections = eval_model(x).data
        print('detections:', detections.size()) # TODO [1, 21, 200, 5]

        cv_img = image
        #TODO 遍历每一个类别
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            #TODO 过滤掉哪些低置信度的框
            mask = dets[:,0].gt(conf_thresh).expand(5,dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1,5)
            #TODO 判断当前的检测结果是否包含框
            if dets.shape[0]==0:
                continue
            if j:
                boxes = dets[:,1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:,0].cpu().numpy()
                for box, score in zip(boxes,scores):
                    x1,y1,x2,y2 = box
                    y1 = int(y1)
                    x1 = int(x1)
                    y2 = int(y2)
                    x2 = int(x2)

                    cv2.rectangle(img=cv_img, pt1=(x1, y1), pt2=(x2, y2),
                                  color=(255, 0, 255), thickness=2, lineType=2)
                    # cv2.putText(cv_img, cfg.VOC_CLASSES[int(j)]+"%.2f"%score, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    #             (255, 0, 255))

                    text = "{} {}%".format(VOC_CLASSES[int(j) - 1], round(score.item() * 100, 2))
                    # cvzone.putTextRect(
                    #     img=cv_img, text=text, pos=(x1 + 9, y1 - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
                    #     font=cv2.FONT_HERSHEY_SIMPLEX
                    # )
        end_time = time.time()
        print('{} inference time: {} seconds'.format(img_name,end_time - start_time))
        cv2.imwrite(os.path.join(save,img_name),cv_img)


def timeDetect():

    cap = cv2.VideoCapture(0)
    count = 0
    start_time = time.time()

    while cap.isOpened():
        ret,frame = cap.read()
        count += 1
        if ret == False:
            break

        frame = cv2.resize(frame,dsize=(800,600))
        frame = cv2.flip(src=frame, flipCode=2)
        h, w = np.shape(frame)[:2]
        scale = torch.Tensor([w, h, w, h])



        with torch.no_grad():
            img = transform(frame)[0]
            img = img[:, :, (2, 1, 0)]  # TODO 转换为RGB
            x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            x = x.to(device)
            if device:
                x = x.to(device)
                scale = scale.to(device)
        # TODO 输出定位框和score分数
        detections = eval_model(x).data
        FPS = int(count / (time.time() - start_time))

        cv_img = frame
        # TODO 遍历每一个类别
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            # TODO 过滤掉哪些低置信度的框
            mask = dets[:, 0].gt(conf_thresh).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            # TODO 判断当前的检测结果是否包含框
            if dets.shape[0] == 0:
                continue
            if j:
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box
                    y1 = int(y1)
                    x1 = int(x1)
                    y2 = int(y2)
                    x2 = int(x2)

                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
                    cv2.putText(img=cv_img, text=str(int(FPS)), org=(50, 50),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                                color=(0, 255, 0),thickness=2)

                    text = "{} {}%".format(VOC_CLASSES[int(j) - 1], round(score.item() * 100, 2))
                    # cvzone.putTextRect(
                    #     img=cv_img, text=text, pos=(x1 + 9, y1 - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
                    #     font=cv2.FONT_HERSHEY_SIMPLEX
                    # )
        cv2.imshow('img',cv_img)
        key = cv2.waitKey(1)
        if key & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    predictImage()
    # timeDetect()
    pass