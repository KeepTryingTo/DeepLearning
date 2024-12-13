'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''
import os
import cv2
import time
import torch
# import cvzone
import numpy as np
from PIL import Image
from torchvision import transforms
from models.backbone import EfficientDetBackbone

from models.efficientdet.augmentations import *
from configs.config import *

from models.efficientdet.utils import BBoxTransform, ClipBoxes
from utiles.utils import postprocess

from models.efficientdet.voc import VOC_CLASS_LIST

#TODO 根据anchor box，对预测box进行解码
regressBoxes = BBoxTransform()
#TODO 对解码之后的预测框进行边界的处理
clipBoxes = ClipBoxes()

def preprocess_img(image,input_ksize):
    '''
    resize image and bboxes
    Returns
    image_paded: input_ksize
    bboxes: [None,4]
    '''
    min_side, max_side    = input_ksize
    h,  w, _  = image.shape

    smallest_side = min(w,h)
    largest_side=max(w,h)
    scale=min_side/smallest_side
    if largest_side*scale>max_side:
        scale=max_side/largest_side
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w=32-nw%32
    pad_h=32-nh%32

    image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.float32)
    image_paded[:nh, :nw, :] = image_resized
    return image_paded

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']


def loadModel(params,resum = None):
    model = EfficientDetBackbone(
        num_classes=len(params.obj_list),
        compound_coef=0,
        ratios=eval(params.anchors_ratios),
        scales=eval(params.anchors_scales),
        load_weights=False
    )
    #TODO 加载之前已经未完全训练完而保存的模型
    if resum is not None:
        checkpoint = torch.load(resum,map_location='cpu')
        # optimizer = checkpoint['optimizer']
        # start_epoch = checkpoint['epoch']
        # model.load_state_dict(checkpoint['model'])
        model.load_state_dict(checkpoint)
    return model

import yaml
class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
params = Params(f'configs/coco.yml')
weight_path = r'./logs/coco/efficientdet-d0_2_4500.pth'

transform=transforms.Compose(
    [
        transforms.Resize(size=(512,512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ]
)

def predictImage(threshold=0.15, iou_threshold=0.15):
    model = loadModel(params=params,resum=weight_path)
    model = model.to(device)
    model.eval()

    root = "images/"
    names = os.listdir(root)
    for name in names:
        img_rgb = Image.open(os.path.join(root,name))
        width,height = img_rgb.size
        img = transform(img_rgb)
        img = img.unsqueeze(dim = 0)
        scale = np.array([width,height,width,height])

        start_t = time.time()
        with torch.no_grad():
            img = img.to(device)
            _, regression, classification, anchors = model(img)

        # TODO 进行后处理，过滤掉那些冗余的框
        out = postprocess(x=img,
                          anchors=anchors, regression=regression,
                          classification=classification,
                          regressBoxes=regressBoxes, clipBoxes=clipBoxes,
                          threshold=threshold, iou_threshold=iou_threshold)

        # TODO [[xmin,ymin,xmax,ymax],...]
        reg_boxes = out[0]['rois']
        if len(reg_boxes) == 0:
            continue
        reg_boxes = reg_boxes[...] / 512
        boxes = reg_boxes * scale
        scores = out[0]['scores']
        classes = out[0]['class_ids']
        print('boxes.shape: {}'.format(np.shape(boxes)))

        img_bgr = cv2.cvtColor(np.array(img_rgb),cv2.COLOR_RGB2BGR)
        for i, box in enumerate(boxes):
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(img_bgr, pt1, pt2, (255, 0, 255),thickness=2)
            # img_pad=cv2.putText(img_pad,"%s %.3f"%(COCODataset.CLASSES_NAME[int(classes[i])],scores[i]),
            #                     (int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,[0,200,20],2)
            # cvzone.putTextRect(img=img_bgr, text="%s %.3f" % (VOC_CLASS_LIST[int(classes[i])], scores[i]),
            #                    pos=(int(box[0]), int(box[1]) - 10), scale=1, thickness=1, colorT=[0, 200, 20])
        cv2.imwrite("runs/" + name, img_bgr)
        end_t = time.time()
        cost_t = 1000 * (end_t - start_t)
        print("===>success detect img, cost time %.2f ms" % cost_t)
        # cv2.imshow('img',img_pad)
        # cv2.waitKey(0)
    cv2.destroyAllWindows()

def timDetect():
    model = loadModel(params,weight_path)
    model.eval()
    cap = cv2.VideoCapture(0)

    while True:
        ret,frame = cap.read()
        frame = cv2.resize(frame,dsize=(800,600))
        height,width,_ = frame.shape
        img = transform(frame)
        img = img.unsqueeze(dim=0)
        scale = np.array([width, height, width, height])

        start_t = time.time()
        with torch.no_grad():
            _, regression, classification, anchors = model(img)
        end_t = time.time()
        cost_t = 1000 * (end_t - start_t)
        print("===>success processing img, cost time %.2f ms" % cost_t)
        # TODO 进行后处理，过滤掉那些冗余的框
        out = postprocess(x=img,
                          anchors=anchors, regression=regression,
                          classification=classification,
                          regressBoxes=regressBoxes, clipBoxes=clipBoxes,
                          threshold=0.005, iou_threshold=0.45)

        # TODO [[xmin,ymin,xmax,ymax],...]
        reg_boxes = out[0]['rois']
        reg_boxes = reg_boxes[...] / 512
        boxes = reg_boxes * scale
        scores = out[0]['scores']
        classes = out[0]['class_ids']

        for i, box in enumerate(boxes):
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0))
            # img_pad=cv2.putText(img_pad,"%s %.3f"%(COCODataset.CLASSES_NAME[int(classes[i])],scores[i]),
            #                     (int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,[0,200,20],2)
            # cvzone.putTextRect(img=frame, text="%s %.3f" % (VOC_CLASS_LIST[int(classes[i])], scores[i]),
            #                    pos=(int(box[0]), int(box[1]) - 10), scale=1, thickness=1, colorT=[0, 200, 20])
        cv2.imshow('img',frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    predictImage()
    # timDetect()
    pass






