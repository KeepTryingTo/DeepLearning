import pickle

import torch
import torch.utils.data as data
from data import *
from utils.trainutils import Timer

import argparse
import os
from models.PFPNetR import build_pfp

def _valid(model, args, iters, dataset,device):
    num_images = len(dataset)
    labelmap = VOC_CLASSES
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]
    if not os.path.exists(os.path.join(args.eval_folder, 'valid')):
        os.mkdir(os.path.join(args.eval_folder, 'valid'))
    det_file = os.path.join(args.eval_folder, 'valid', 'detection_%d.pkl' %iters )

    data_loader = data.DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=0,
                                  shuffle=False,
                                  pin_memory=False)
    timer = Timer('m')
    timer.tic()
    for i, batch_data in enumerate(data_loader):
        images, gt, height, width = batch_data
        images = images.to(device)

        with torch.no_grad():
            detections = model(images).data

        for j in range(1, detections.size(1)):

            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)

            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= width
            boxes[:, 2] *= width
            boxes[:, 1] *= height
            boxes[:, 3] *= height
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets
        print('\r%04d'%i, end='')

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    mAP = dataset.validate_detections(all_boxes, os.path.join(args.eval_folder, 'valid'))
    model.train()
    print('\r%d iters, take %.2f mins'%(iters, timer.toc()))
    return mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--model_name', default='PFPNetR', type=str)

    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'], type=str, help='VOC or COCO')
    parser.add_argument('--input_size', default='320', choices=['320', '512'], type=str)
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')

    # Test args
    parser.add_argument('--eval_folder', default='./eval/', type=str, help='Directory for the output of evaluation')
    parser.add_argument('--confidence_threshold', default=0.01, type=float, help='Detection confidence threshold')
    parser.add_argument('--top_k', default=200, type=int, help='Further restrict the number of predictions to parse')
    args = parser.parse_args()

    # VOC_ROOT = r'D:\conda3\Transfer_Learning\PASCAL_VOC\VOCdevkit'
    dataset = VOCDetection(
         root=VOC_ROOT, mode='test',
         image_sets=[('2007', 'test')],
         transform=BaseTransform(args.input_size,
                                 mean=(104, 117, 123)
                                 )
    )

    weight_path = r'weights/PFPNetR320.pth'
    # TODO 加载预训练模型用于验证
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg = data_configs['VOC']['320']
    eval_model = build_pfp('test',
                           cfg['min_dim'],
                           cfg['num_classes'],
                           device=device)
    eval_model.load_state_dict(torch.load(weight_path,map_location='cpu')['model'])
    eval_model.eval()
    eval_model.to(device)
    print('Finished loading model!')
    print('Evaluating on test and size: {}'.format(len(dataset)))

    _valid(eval_model,args,0,dataset,device)

"""
VOC2012 trainval.txt and VOC2007 trainval.txt,finally test on 2007 test.txt
    iters = 115000 Mean AP = 0.7180
    

"""