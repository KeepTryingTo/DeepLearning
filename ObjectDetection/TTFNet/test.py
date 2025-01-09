import os
import json
import torch
import cv2 as cv
import numpy as np
from tqdm import tqdm
import yaml
from commons.augmentations import ScalePadding
from nets.centernet import CenterNet

rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

coco_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48,
            49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90]




def write_coco_json():
    from pycocotools.coco import COCO
    device = torch.device("cuda:0")
    img_root = "/home/ff/myProject/KGT/myProjects/myDataset/coco/val2014/val2014"

    with open("config/centernet.yaml", 'r') as rf:
        cfg = yaml.safe_load(rf)
    model_cfg = cfg['model']
    model = CenterNet(num_cls=model_cfg['num_cls'],
                      PIXEL_MEAN=model_cfg['PIXEL_MEAN'],
                      PIXEL_STD=model_cfg['PIXEL_STD'],
                      backbone=model_cfg['backbone'],
                      cfg=model_cfg)

    weights = torch.load("weights/0_TTFNet_best_map.pth",map_location='cpu')['ema']
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    basic_transform = ScalePadding(target_size=(512, 512), padding_val=(103, 116, 123))
    coco = COCO("/home/ff/myProject/KGT/myProjects/myDataset/coco/captions/annotations/instances_val2014.json")
    coco_predict_list = list()
    for img_id in tqdm(coco.imgs.keys()):
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(img_root, file_name)
        img = cv.imread(img_path)
        # ori_img = img.copy()
        img, ratio, (left, top) = basic_transform.make_border(img)
        h, w = img.shape[:2]
        img_out = img[:, :, [2, 1, 0]].astype(np.float32) / 255.0
        img_out = ((img_out - np.array(rgb_mean)) / np.array(rgb_std)).transpose(2, 0, 1).astype(np.float32)
        img_out = torch.from_numpy(np.ascontiguousarray(img_out)).unsqueeze(0).float().to(device)
        predicts = model(img_out)  #list(predict) predict.shape=[num_box,6]  6==>x1,y1,x2,y2,score,label

        for i in range(len(predicts)):
            predicts[i][:, [0, 2]] = predicts[i][:, [0, 2]].clamp(min=0, max=w)
            predicts[i][:, [1, 3]] = predicts[i][:, [1, 3]].clamp(min=0, max=h)

        box = predicts[0]
        if box is None:
            continue
        box[:, [0, 2]] = (box[:, [0, 2]] - left) / ratio[0]
        box[:, [1, 3]] = (box[:, [1, 3]] - top) / ratio[1]
        box = box.detach().cpu().numpy()
        # ret_img = draw_box(ori_img, box[:, [4, 5, 0, 1, 2, 3]], colors=coco_colors)
        # cv.imwrite(file_name, ret_img)
        coco_box = box[:, :4]
        coco_box[:, 2:] = coco_box[:, 2:] - coco_box[:, :2]
        for p, b in zip(box.tolist(), coco_box.tolist()):
            coco_predict_list.append({'image_id': img_id,
                                      'category_id': coco_ids[int(p[5])],
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5)})
    with open("predicts.json", 'w') as file:
        json.dump(coco_predict_list, file)


def coco_eavl():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO("/home/ff/myProject/KGT/myProjects/myDataset/coco/captions/annotations/instances_val2014.json")  # initialize COCO ground truth api

    cocoDt = cocoGt.loadRes("predicts.json")  # initialize COCO pred api
    imgIds = [img_id for img_id in cocoGt.imgs.keys()]
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds  # image IDs to evaluate
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    write_coco_json()
    coco_eavl()
"""
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.276
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.433
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.294
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.100
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.293
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.381
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.424
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.210
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.630
 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.303
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.470
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.324
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.111
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.321
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.425
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.278
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.443
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.463
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.225
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.652
"""