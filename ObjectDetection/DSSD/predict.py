"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/2/26 18:11
"""

import os
import cv2
import cvzone
import time
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
from models.prior_box import PriorBox
from models.inference import PostProcessor
from models.dssd_detector import DSSDDetector,createCfg
from utiles import box_utils
from dataset.myDir.transforms import Resize,SubtractMeans,ToTensor,Compose

device = 'cpu' if torch.cuda.is_available() else 'cpu'
save = r'./outputs'
cfg = createCfg(config_file=r'configs/resnet101_dssd320_voc0712.yaml')
checkpoint = torch.load(r'weights/voc_325_dssd.pth.tar',map_location='cpu')['model']
model = DSSDDetector(cfg)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
print('load model is done ...')
print('using device is {}'.format(device))

VOC_CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

transform = Compose(
    [
        Resize(cfg.INPUT.IMAGE_SIZE),
        SubtractMeans(cfg.INPUT.PIXEL_MEAN),
        ToTensor()
    ]
)

def detect(bbox_pred,cls_logits):
    priors = PriorBox(cfg)().to(bbox_pred.device)
    scores = F.softmax(cls_logits, dim=2)
    """
        CENTER_VARIANCE: 0.1 
        SIZE_VARIANCE: 0.2
    """
    boxes = box_utils.convert_locations_to_boxes(
        bbox_pred, priors, cfg.MODEL.CENTER_VARIANCE, cfg.MODEL.SIZE_VARIANCE
    )
    # 将box的[x,y,w,h] => [xmin,ymin,xmax,ymax]
    boxes = box_utils.center_form_to_corner_form(boxes)
    # 得到检测结果
    detections = (scores, boxes)
    # 进行NMS处理
    detections = PostProcessor(detections)
    return detections, {}

def detectSingleImage(conf_threshold = 0.25):
    root = r'images'
    with torch.no_grad():
        for imgName in os.listdir(root):
            start_time = time.time()

            img_path = os.path.join(root,imgName)
            img = Image.open(img_path)
            width, height = img.size
            if img.convert("L"):
                img = img.convert('RGB')

            img = np.array(img)
            image,_,_ = transform(img,None,None)
            image = image.unsqueeze(0)
            detections,_,_ = model(image)

            detections = detections[0]
            boxes = detections['boxes']
            labels = detections['labels']
            scores = detections['scores']

            #TODO 首先将坐标[xmin,ymin,xmax,ymax]缩放至[0-1]之间
            boxes[:, 0::2] /= cfg.INPUT.IMAGE_SIZE
            boxes[:, 1::2] /= cfg.INPUT.IMAGE_SIZE
            #TODO 按照相对原图大小进行缩放
            boxes[:, 0::2] *= width
            boxes[:, 1::2] *= height

            print('pred_boxes: ',detections['boxes'].size())
            print('pred_classes: ',detections['labels'].size())
            print('pred_scores: ',detections['scores'].size())

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for i,(box, score) in enumerate(zip(boxes, scores)):
                x1,y1,x2,y2 = box
                y1 = int(y1)
                x1 = int(x1)
                y2 = int(y2)
                x2 = int(x2)

                label = labels[i]

                if score > conf_threshold:
                    cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2),
                                  color=(255, 0, 255), thickness=2, lineType=2)
                    # cv2.putText(img, cfg.VOC_CLASSES[int(j)]+"%.2f"%score, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    #             (255, 0, 255))

                    text = "{} {}%".format(VOC_CLASSES[int(label)], round(score.item() * 100, 2))
                    cvzone.putTextRect(
                        img=img, text=text, pos=(x1 + 9, y1 - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
                        font=cv2.FONT_HERSHEY_SIMPLEX
                    )
            end_time = time.time()
            print('{} inference time: {} seconds'.format(imgName, end_time - start_time))
            cv2.imwrite(os.path.join(save, imgName), img)

def timeDetect(conf_threshold = 0.5):

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

        with torch.no_grad():
            img = Image.fromarray(frame)
            x, _, _ = transform(img, None, None)
            x = x.unsqueeze(0)
            if device:
                x = x.to(device)
                scale = scale.to(device)
        # TODO 输出定位框和score分数
        detections = model(x)
        boxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']

        FPS = int(count / (time.time() - start_time))
        cv_img = frame

        for i,(box, score) in enumerate(zip(boxes, scores)):
            x1,y1,x2,y2 = box
            y1 = int(y1)
            x1 = int(x1)
            y2 = int(y2)
            x2 = int(x2)

            label = labels[i]

            if score > conf_threshold:
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
                cv2.putText(img=cv_img, text=str(int(FPS)), org=(50, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                            color=(0, 255, 0),thickness=2)

                text = "{} {}%".format(VOC_CLASSES[int(label)], round(score.item() * 100, 2))
                cvzone.putTextRect(
                    img=cv_img, text=text, pos=(x1 + 9, y1 - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
                    font=cv2.FONT_HERSHEY_SIMPLEX
                )
        cv2.imshow('img',cv_img)
        key = cv2.waitKey(1)
        if key & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detectSingleImage()
    # timeDetect()
    pass

