"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/10/31-18:52
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import warnings
warnings.filterwarnings('ignore')
import pickle
from m2det import build_net
from layers.functions import Detect,PriorBox
from data import BaseTransform
from utils.core import *

from utils.timer import Timer
from torchvision.ops import nms

parser = argparse.ArgumentParser(description='M2Det Testing')
parser.add_argument('-c', '--config', default='configs/m2det320_vgg.py', type=str)
parser.add_argument('-d', '--dataset', default='VOC',help='VOC or COCO version')
parser.add_argument('-m', '--trained_model',
                    default=r'weights/M2Det_VOC_size320_netvgg16_epoch120.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--test', action='store_true',help='to submit a test file')
parser.add_argument('--retest', default=False, type=bool,help='test cache results')
args = parser.parse_args()

print_info('----------------------------------------------------------------------\n'
           '|                       M2Det Evaluation Program                     |\n'
           '----------------------------------------------------------------------',
           ['yellow','bold'])

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
cfg = Config.fromfile(args.config)
if not os.path.exists(cfg.test_cfg.save_folder):
    os.mkdir(cfg.test_cfg.save_folder)
anchor_config = anchors(cfg)
print_info('The Anchor info ==================================> ')
for k, v in anchor_config.items():
    print('{} : {}'.format(k,v))
print_info('=============== ==================================> ')

priorbox = PriorBox(anchor_config,device = device)
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(device)

#TODO 对VOC07 test进行验证
def test_net(save_folder, net,
             detector, cuda,
             testset, transform,
             max_per_image=300,
             iou_thresh=0.5, conf_threshod=0.2):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    num_classes = (21, 81)[args.dataset == 'COCO']
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return

    # TODO 对所有的图像进行检测
    for i in range(num_images):
        img = testset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            x = x.to(device)
            scale = scale.to(device)

        _t['im_detect'].tic()
        out = net(x)  # forward pass
        boxes, scores = detector.forward(out, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]
        boxes *= scale

        _t['misc'].tic()
        # TODO 针对每一个类别都进行遍历
        for j in range(1, num_classes):
            inds = torch.where(scores[:, j] > conf_threshod)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            # TODO 得到当前类别的预测框以及scores分数
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            keep = nms(c_bboxes, c_scores, iou_threshold=iou_thresh)

            c_bboxes = c_bboxes[keep]
            c_scores = c_scores[keep]
            c_bboxes = c_bboxes.cpu().numpy()
            c_scores = c_scores.cpu().numpy()

            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                  .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)


if __name__ == '__main__':
    net = build_net('test',
                    size=cfg.model.input_size,
                    config=cfg.model.m2det_config)
    init_net(net, cfg, args.trained_model)
    print_info('===> Finished constructing and loading model',
               ['yellow', 'bold'])
    net.eval()
    _set = 'eval_sets' if not args.test else 'test_sets'
    testset = get_dataloader(cfg, args.dataset, _set)
    print("Test dataset size: {}".format(len(testset)))

    net = net.to(device)

    detector = Detect(cfg.model.m2det_config.num_classes,
                      cfg.loss.bkg_label, anchor_config, device=device)

    save_folder = os.path.join(cfg.test_cfg.save_folder, args.dataset)
    _preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
    test_net(save_folder=save_folder,
             net=net,
             detector=detector,
             cuda=device,
             testset=testset,
             transform=_preprocess,
             max_per_image=cfg.test_cfg.topk,
             iou_thresh=0.005,
             conf_threshod=0.005)

"""

('2007', 'trainval'), ('2012', 'trainval') and test on 2007 test.txt for vgg
    iou_thresh=0.005,conf_threshod = 0.005 Mean AP = 0.6392
"""