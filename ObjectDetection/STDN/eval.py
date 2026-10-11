"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/11/10-19:02
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import os.path as osp
import numpy as np
import torch
import pickle
from utils.utils import to_var
from model import STDN
from layers.anchor_box import AnchorBox
from utils.timer import Timer

from data.pascal_voc import save_results as voc_save, do_python_eval

from torch.utils.data import DataLoader
from data.pascal_voc import PascalVOC
from data.augmentations import Augmentations, BaseTransform


VOC_CONFIG = {
    '0712': ([('2007', 'trainval'), ('2012', 'trainval')],
             [('2007', 'test')])
}

def detection_collate(batch):
    images = []
    targets = []
    for sample in batch:
        images.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(images, 0), targets

def eval(dataset, top_k, threshold,
         result_save_path,
         pretrained_model = 'stdn_densenet169'):
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(21)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    results_path = osp.join(result_save_path,
                            pretrained_model)
    det_file = osp.join(results_path,
                        'detections.pkl')

    detect_times = []

    with torch.no_grad():
        for i in range(num_images):
            image, target, h, w = dataset.pull_item(i)
            image = image.unsqueeze(0).to(device=device)

            _t['im_detect'].tic()
            detections = model(image).data
            detect_time = _t['im_detect'].toc(average=False)
            detect_times.append(detect_time)

            # skip j = 0 because it is the background class
            for j in range(1, detections.shape[1]):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.shape[0] == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                all_boxes[j][i] = cls_dets

            print('im_detect: {:d}/{:d} {:.3f}'.format(i + 1,
                                                       num_images,
                                                       detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')

    if datasetName == 'voc':
        voc_save(all_boxes, dataset, results_path)
        do_python_eval(results_path, dataset)

    detect_times = np.asarray(detect_times)
    detect_times.sort()
    print('fps[0500]:', (1 / np.mean(detect_times[:500])))
    print('fps[1000]:', (1 / np.mean(detect_times[:1000])))
    print('fps[1500]:', (1 / np.mean(detect_times[:1500])))
    print('fps[2000]:', (1 / np.mean(detect_times[:2000])))
    print('fps[2500]:', (1 / np.mean(detect_times[:2500])))
    print('fps[3000]:', (1 / np.mean(detect_times[:3000])))
    print('fps[3500]:', (1 / np.mean(detect_times[:3500])))
    print('fps[4000]:', (1 / np.mean(detect_times[:4000])))
    print('fps[4500]:', (1 / np.mean(detect_times[:4500])))
    print('fps[all]:', (1 / np.mean(detect_times)))





if __name__ == '__main__':
    # TODO instatiate anchor boxes(SSD)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    anchor_boxes = AnchorBox(map_sizes=[1, 3, 5, 9, 18, 36],
                             aspect_ratios=[1.6, 2, 3])

    anchor_boxes = anchor_boxes.get_boxes()
    anchor_boxes = anchor_boxes.to(device)

    x = torch.zeros(size=(1, 3, 300, 300))
    model = STDN(
        mode='test',
        stdn_config='300',
        channels=3,
        class_count=21,
        anchors=anchor_boxes,
        num_anchors=8,
        new_size=300
    )
    weight_path = r'./models/2024-11-14 11_43_58.358238/340000.pth'
    model.load_state_dict(torch.load(weight_path,map_location='cpu'))
    model.eval()
    model.to(device)
    print('load model is done...')

    image_transform = BaseTransform(300, (104, 117, 123))
    voc_root = r'/home/ff/myProject/KGT/myProjects/myDataset/voc2012'
    test_dataset = PascalVOC(data_path=voc_root,
                             image_sets=[('2007', 'test')],
                             new_size=300,
                             mode='test',
                             image_transform=image_transform)
    # test_loader = DataLoader(dataset=test_dataset,
    #                          batch_size=1,
    #                          shuffle=False,
    #                          collate_fn=detection_collate,
    #                          num_workers=4,
    #                          pin_memory=True)
    print('load dataset is done...')
    datasetName = 'voc'
    eval(dataset=test_dataset,
            top_k=100,
            threshold=0.01,
            result_save_path=r'./results/')

"""
train on: VOC07 trainval.txt and VOC12 trainval.txt
test on: VOC07 test.txt
Mean AP = 0.6375
"""