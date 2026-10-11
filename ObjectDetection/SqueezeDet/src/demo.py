import os
import glob
import tqdm
import cv2
import time

import numpy as np
# import skimage.io
import torch
import torch.utils.data
from PIL import Image

from datasets.kitti import KITTI
from engine.detector import Detector
from model.squeezedet import SqueezeDet
from utils.model import load_model
from torchvision import transforms

from utils.config import Config
from utils.misc import init_env

from src.datasets.voc0712 import VOCDetection
from src.datasets.data_augment import preproc,BaseTransform

save_dir = r'/data1/KTG/myProject/SqueezeDet-PyTorch-master/outputs'
# class_names = ('Car', 'Pedestrian', 'Cyclist')

class_names = (
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

transform = transforms.Compose([
    transforms.Resize(size=(512, 768)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[93.877, 98.801, 95.923],
        std=[78.782, 80.130, 81.200]
    )
])

def demo(cfg):
    # prepare configurations
    cfg.load_model = '/data1/KTG/myProject/SqueezeDet-PyTorch-master/models/squeezedet_kitti_epoch280.pth'
    # cfg.load_model = r'/data1/KTG/myProject/SqueezeDet-PyTorch-master/runs/model_resize_image_best.pth'
    cfg.debug_dir = save_dir
    cfg.gpus = [-1]  # -1 to use CPU
    cfg.debug = 2  # to visualize detection boxes
    dataset = KITTI('val', cfg)
    cfg = Config().update_dataset_info(cfg, dataset)

    # preprocess image to match model's input resolution
    preprocess_func = dataset.preprocess
    del dataset

    # prepare model & detector
    model = SqueezeDet(cfg)
    model = load_model(model, cfg.load_model)
    detector = Detector(model.to(cfg.device), cfg)

    # prepare images
    # sample_images_dir = '../data/samples/kitti/testing/image_2'
    sample_images_dir = r'/data1/KTG/myProject/SqueezeDet-PyTorch-master/images'
    sample_image_paths = glob.glob(os.path.join(sample_images_dir, '*.png'))

    # detection
    for path in tqdm.tqdm(sample_image_paths):
        # image = skimage.io.imread(path).astype(np.float32)
        image_bgr = cv2.imread(path)
        # image_bgr = cv2.resize(image_bgr,dsize=(1248,384))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        #TODO 保存图像的ID以及原始图像大小
        image_meta = {'image_id': os.path.basename(path)[:-4],
                      'orig_size': np.array(image_bgr.shape[:2], dtype=np.int32)}

        image, image_meta, _ = preprocess_func(image_rgb, image_meta)
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(cfg.device)
        image_meta = {k: torch.from_numpy(v).unsqueeze(0).to(cfg.device) if isinstance(v, np.ndarray)
                      else [v] for k, v in image_meta.items()}

        inp = {'image': image,
               'image_meta': image_meta}

        resuult = detector.detect(inp)


class_colors = (255. * np.array(
    [0.850, 0.325, 0.098,
     0.466, 0.674, 0.188,
     0.098, 0.325, 0.850,
     0.301, 0.745, 0.933,
     0.635, 0.078, 0.184,
     0.300, 0.300, 0.300,
     0.600, 0.600, 0.600,
     1.000, 0.000, 0.000,
     1.000, 0.500, 0.000,
     0.749, 0.749, 0.000,
     0.000, 1.000, 0.000,
     0.000, 0.000, 1.000,
     0.667, 0.000, 1.000,
     0.333, 0.333, 0.000,
     0.333, 0.667, 0.000,
     0.333, 1.000, 0.000,
     0.667, 0.333, 0.000,
     0.667, 0.667, 0.000,
     0.667, 1.000, 0.000,
     1.000, 0.333, 0.000,
     1.000, 0.667, 0.000,
     1.000, 1.000, 0.000,
     0.000, 0.333, 0.500,
     0.000, 0.667, 0.500,
     0.000, 1.000, 0.500]
)).astype(np.uint8).reshape((-1, 3))


def predictImage():
    cfg = Config().parse()
    init_env(cfg)
    root = r'/data1/KTG/myProject/SqueezeDet-PyTorch-master/images'
    # cfg.load_model = r'/data1/KTG/myProject/SqueezeDet-PyTorch-master/models/squeezedet_kitti_epoch280.pth'
    cfg.load_model = r'/data1/KTG/myProject/SqueezeDet-PyTorch-master/runs/voc_model_2.944_best.pth'
    cfg.debug_dir = save_dir

    # cfg.load_model = '/data1/KTG/myDataset/KITTI/exp/default/model_best.pth'
    cfg.gpus = [-1]  # -1 to use CPU
    cfg.debug = 2  # to visualize detection boxes
    root_dir = r'/data1/KTG/myDataset/VOC'
    VOC = dict(
        train_sets=[('2007', 'trainval'), ('2012', 'trainval')],
        eval_sets=[('2007', 'test')],
    )
    img_size = (512, 768)
    rgb_means = (103.94, 116.78, 123.68)
    p = 0.6
    _preproc = preproc(resize=img_size,
                       rgb_means=rgb_means,
                       p=p)
    valDataset = VOCDetection(
        img_size=img_size,
        root=root_dir, image_sets=VOC['eval_sets'],
        preproc=_preproc
    )
    cfg = Config().update_dataset_info(cfg, valDataset)

    transform = BaseTransform(
        resizes=img_size,
        rgb_means=rgb_means
    )


    model = SqueezeDet(cfg)
    model = load_model(model, cfg.load_model).to(cfg.device)
    detector = Detector(model, cfg)

    for imgName in os.listdir(root):
        start_time = time.time()
        imgPath = os.path.join(root, imgName)
        image_bgr = cv2.imread(imgPath)
        height,width,_ = image_bgr.shape

        PIL_Image = Image.fromarray(image_bgr)
        img = transform(PIL_Image).unsqueeze(0)
        img = img.to(cfg.device)
        dict_image = {'image': img}
        with torch.no_grad():
            dets = model(dict_image)
            batch_size = dets['class_ids'].shape[0]
            for b in range(batch_size):
                det = {k: v[b] for k, v in dets.items()}
                det = detector.filter(det)

                if det is None:
                    continue

                det = {k: v.cpu().numpy() for k, v in det.items()}
        if det is None:
            continue
        class_ids,boxes,scores = det['class_ids'], det['boxes'], det['scores']
        print('det.shape: {}'.format(np.shape(boxes)))
        #TODO 首先将其缩放至0-1之间
        boxes[:, [0, 2]] /= img_size[1]
        boxes[:, [1, 3]] /= img_size[0]
        #TODO 然后缩放至原图大小
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        num_boxes = boxes.shape[0]
        for i in range(num_boxes):
            if scores[i] < cfg.score_thresh:
                continue
            # TODO 预测的类别以及box
            class_id = class_ids[i]
            bbox = boxes[i].astype(np.uint32).tolist()
            # TODO 坐标框绘制到图像上
            image = cv2.rectangle(image_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  class_colors[class_id].tolist(), 2)

            class_name = class_names[class_id] if class_names is not None else 'class_{}'.format(class_id)
            text = '{} {:.2f}'.format(class_name, scores[i]) if scores is not None else class_name
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, fontScale=.5, thickness=1)[0]
            cv2.rectangle(image,
                                  (bbox[0], bbox[1] - text_size[1] - 8),
                                  (bbox[0] + text_size[0] + 8, bbox[1]),
                                  class_colors[class_id].tolist(), -1)
            cv2.putText(image, text, (bbox[0] + 4, bbox[1] - 4), font,
                                fontScale=.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


        end_time = time.time()
        cv2.imwrite(os.path.join(r'/data1/KTG/myProject/SqueezeDet-PyTorch-master/outputs/',imgName) ,image_bgr)
        print('detect {} is finished and time is: {}'.format(imgName,end_time- start_time))


# def timeDetect(conf_threshold = 0.5):
#     cap = cv2.VideoCapture(0)
#     count = 0
#     start_time = time.time()
#
#     cfg = Config().parse()
#     init_env(cfg)
#
#     cfg.load_model = r'/data1/KTG/myProject/SqueezeDet-PyTorch-master/models/squeezedet_kitti_epoch280.pth'
#     # cfg.load_model = '/data1/KTG/myProject/SqueezeDet-PyTorch-master/runs/model_best.pth'
#     cfg.gpus = [-1]  # -1 to use CPU
#     cfg.debug = 2  # to visualize detection boxes
#     dataset = KITTI('val', cfg)
#     cfg = Config().update_dataset_info(cfg, dataset)
#
#     # preprocess image to match model's input resolution
#     preprocess_func = dataset.preprocess
#     del dataset
#
#     model = SqueezeDet(cfg)
#     model = load_model(model, cfg.load_model)
#     detector = Detector(model.to(cfg.device), cfg)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         count += 1
#         if ret == False:
#             break
#
#         frame = cv2.resize(frame, dsize=(800, 600))
#         frame = cv2.flip(src=frame, flipCode=2)
#         h, w = np.shape(frame)[:2]
#
#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
#
#         image_meta = {'image_id': "",
#                       'orig_size': np.array(frame.shape[:2], dtype=np.int32)}
#
#         image, image_meta, _ = preprocess_func(image_rgb, image_meta)
#         image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(cfg.device)
#         image_meta = {k: torch.from_numpy(v).unsqueeze(0).to(cfg.device) if isinstance(v, np.ndarray)
#         else [v] for k, v in image_meta.items()}
#
#         inp = {'image': image,
#                'image_meta': image_meta}
#
#         results = detector.detect(inp)
#
#         allboxes = np.array(results)
#         boxes = allboxes['boxes']
#         scores = allboxes['scores']
#         cls_inds = allboxes['class_ids']
#
#         # TODO 遍历每一个类别
#         for i, box in enumerate(boxes):
#             box = [int(_) for _ in box]
#             confidence = scores[i]
#             label = cls_inds[i]
#
#             if confidence > conf_threshold:
#                 x1, y1, x2, y2 = box
#                 y1 = int(y1)
#                 x1 = int(x1)
#                 y2 = int(y2)
#                 x2 = int(x2)
#
#                 cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2),
#                               color=(255, 0, 255), thickness=2, lineType=2)
#                 # cv2.putText(cv_img, cfg.VOC_CLASSES[int(label)]+"%.2f"%score, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                 #             (255, 0, 255))
#
#                 # text = "{} {}%".format(VOC_CLASSES[int(label) - 1], round(score.item() * 100, 2))
#                 # cvzone.putTextRect(
#                 #     img=cv_img, text=text, pos=(x1 + 9, y1 - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
#                 #     font=cv2.FONT_HERSHEY_SIMPLEX
#                 # )
#         cv2.imshow('img', frame)
#         key = cv2.waitKey(1)
#         if key & 0xff == 27:
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()


if __name__ == '__main__':
    cfg = Config().parse()
    init_env(cfg)
    # demo(cfg)
    predictImage()
    pass