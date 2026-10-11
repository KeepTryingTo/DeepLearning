"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/10/18-19:37
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import cv2
import cvzone
from PIL import Image

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
                    default=r'./weights/M2Det_VOC_size320_netvgg16_epoch120.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--test', action='store_true',help='to submit a test file')
parser.add_argument('--retest', default=False, type=bool,help='test cache results')
parser.add_argument('--save_folder', default='./outputs', type=str,
                        help='Dir to save results')
args = parser.parse_args()

print_info('----------------------------------------------------------------------\n'
           '|                       M2Det Evaluation Program                     |\n'
           '----------------------------------------------------------------------',
           ['yellow','bold'])

device = 'cpu' if torch.cuda.is_available() else 'cpu'
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

VOC_CLASSES = ( '__background__', # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

COCO_CLASSES = (
    '__background__','person','bicycle',
    'car','motorcycle','airplane','bus',
    'train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter',
    'bench','bird','cat','dog','horse','sheep',
    'cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase',
    'frisbee','skis','snowboard','sports ball','kite',
    'baseball bat','baseball glove','skateboard',
    'surfboard','tennis racket','bottle','wine glass',
    'cup','fork','knife','spoon','bowl','banana',
    'apple','sandwich','orange','broccoli','carrot',
    'hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table',
    'toilet','tv','laptop','mouse','remote',
    'keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock',
    'vase','scissors','teddy bear','hair drier',
    'toothbrush'
)

num_classes = len(VOC_CLASSES)
def predictImage(
        save_folder,
        net,
        detector,
        device,
        transform,
        topk = 100,
        iou_thresh=0.5,
        conf_threshod=0.5
):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    root = r"./images"
    img_list = os.listdir(root)

    # TODO 对所有的图像进行检测
    for imgName in img_list:
        start_time = time.time()

        img_path = os.path.join(root,imgName)
        img = Image.open(img_path)
        w,h = img.size
        scale = torch.Tensor([w,h,w,h])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if device:
                x = x.to(device)
                scale = scale.to(device)
        #TODO 输出定位框和score分数
        out = net(x)
        #TODO 对预测结果进行解码，得到相对于当前图像的大小
        #TODO boxes: [batch_size,11620,4] scores: [batch_size,11620,81]
        boxes, scores = detector.forward(out, priors)
        boxes = boxes[0] #TODO [11620,4]
        scores = scores[0] #TODO [11620,81]
        boxes *= scale

        print('before boxes.shape: {}'.format(boxes.size()))
        print('before score.shape: {}'.format(scores.size()))
        print("num classes: {}".format(num_classes))

        #TODO 用于保存一张图像预测的所有类别结果
        outputs = torch.zeros(size=(scores.size()[1],topk,5))
        for j in range(1, num_classes):
            inds = torch.where(scores[:, j] > conf_threshod)[0]
            if len(inds) == 0:
                continue
            # TODO 得到当前类别的预测框以及scores分数
            r_boxes = boxes[inds]
            r_scores = scores[inds, j]

            # TODO 这里再一次进行筛选，只选择score前topk个概率最大的
            r_scores, sorted_index = torch.sort(r_scores, dim=0, descending=True)
            if r_scores.size()[0] > topk:
                r_scores = r_scores[sorted_index[:topk]]
                r_boxes = r_boxes[sorted_index[:topk]]

            #TODO 过滤掉重叠的框
            keep = nms(r_boxes, r_scores, iou_threshold=iou_thresh)
            r_boxes = r_boxes[keep]
            r_scores = r_scores[keep].unsqueeze(dim=1)

            outputs[j,:len(keep),:] = torch.cat([r_boxes, r_scores],dim=1)

        end_time = time.time()
        print('output.shape: {}'.format(outputs.shape))
        print('detect finished {} time is: {}s'.format(imgName, end_time - start_time))

        cv_img = np.array(img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        #TODO 遍历所有的类别，除了背景之外
        for i in range(1, outputs.size()[0]):
            #TODO 获得当前类的所有box和置信度
            boxes = outputs[i,:,:4]
            confidences = outputs[i,:,4]
            for j in range(len(boxes)):
                box = boxes[j]
                confidence = confidences[j]
                #TODO 对于低置信度和背景都过滤掉
                if confidence > conf_threshod:
                    box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                    cv2.rectangle(
                        img=cv_img, pt1=(box[0], box[1]), pt2=(box[2], box[3]),
                        color=(255, 0, 255), thickness=1
                    )
                    if args.dataset == 'VOC':
                        text = "{} {}%".format(VOC_CLASSES[int(i)], round(confidence.item() * 100, 2))
                    else:
                        text = "{} {}%".format(COCO_CLASSES[int(i)], round(confidence.item() * 100, 2))
                    cvzone.putTextRect(
                        img=cv_img, text=text, pos=(box[0] + 9, box[1] - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
                        font=cv2.FONT_HERSHEY_SIMPLEX
                    )

        cv2.imwrite(os.path.join(save_folder, imgName), cv_img)
        # cv2.imshow('img',cv_img)
        # cv2.waitKey(0)

    cv2.destroyAllWindows()

def timeDetect(
    net,
    detector,
    device,
    transform,
    topk = 100,
    iou_thresh=0.5,
    conf_threshod=0.5
):

    cap = cv2.VideoCapture(0)
    count = 0
    start_time = time.time()

    while cap.isOpened():
        frame, ret = cap.read()
        count += 1
        if ret == False:
            break

        frame = cv2.resize(frame, dsize=(800, 600))
        frame = cv2.flip(src=frame, flipCode=2)
        h, w = np.shape(frame)[:2]
        scale = torch.Tensor([w, h, w, h])

        with torch.no_grad():
            img = Image.fromarray(frame)
            x = transform(img).unsqueeze(0)
            if device:
                x = x.to(device)
                scale = scale.to(device)
        # TODO 输出定位框和score分数
        out = net(x)
        # TODO 对预测结果进行解码，得到相对于当前图像的大小
        # TODO boxes: [batch_size,11620,4] scores: [batch_size,11620,81]
        boxes, scores = detector.forward(out, priors)
        boxes = boxes[0]  # TODO [11620,4]
        scores = scores[0]  # TODO [11620,81]
        boxes *= scale

        # print('before boxes.shape: {}'.format(boxes.size()))
        # print('before score.shape: {}'.format(scores.size()))
        # print("num classes: {}".format(num_classes))

        # TODO 用于保存一张图像预测的所有类别结果
        outputs = torch.zeros(size=(scores.size()[1], topk, 5))
        for j in range(1, num_classes):
            inds = torch.where(scores[:, j] > conf_threshod)[0]
            if len(inds) == 0:
                continue
            # TODO 得到当前类别的预测框以及scores分数
            r_boxes = boxes[inds]
            r_scores = scores[inds, j]

            # TODO 这里再一次进行筛选，只选择score前topk个概率最大的
            r_scores, sorted_index = torch.sort(r_scores, dim=0, descending=True)
            if r_scores.size()[0] > topk:
                r_scores = r_scores[sorted_index[:topk]]
                r_boxes = r_boxes[sorted_index[:topk]]

            # TODO 过滤掉重叠的框
            keep = nms(r_boxes, r_scores, iou_threshold=iou_thresh)
            r_boxes = r_boxes[keep]
            r_scores = r_scores[keep].unsqueeze(dim=1)

            outputs[j, :len(keep), :] = torch.cat([r_boxes, r_scores], dim=1)

        FPS = int(count / (time.time() - start_time))

        cv_img = frame

        # TODO 遍历所有的类别，除了背景之外
        for i in range(1, outputs.size()[0]):
            # TODO 获得当前类的所有box和置信度
            boxes = outputs[i, :, :4]
            confidences = outputs[i, :, 4]
            for j in range(len(boxes)):
                box = boxes[j]
                confidence = confidences[j]
                # TODO 对于低置信度和背景都过滤掉
                if confidence > conf_threshod:
                    box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                    cv2.rectangle(
                        img=cv_img, pt1=(box[0], box[1]), pt2=(box[2], box[3]),
                        color=(255, 0, 255), thickness=1
                    )
                    cv2.putText(img=cv_img, text=str(int(FPS)), org=(50, 50),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                                color=(0, 255, 0), thickness=2)
                    if args.dataset == 'VOC':
                        text = "{} {}%".format(VOC_CLASSES[int(i)], round(confidence.item() * 100, 2))
                    else:
                        text = "{} {}%".format(COCO_CLASSES[int(i)], round(confidence.item() * 100, 2))
                    cvzone.putTextRect(
                        img=cv_img, text=text, pos=(box[0] + 9, box[1] - 12), scale=0.5, thickness=1, colorR=(0, 255, 0),
                        font=cv2.FONT_HERSHEY_SIMPLEX
                    )

        cv2.imshow('img',cv_img)
        key = cv2.waitKey(1)
        if key & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    net = build_net('test',
                    size=cfg.model.input_size,
                    config=cfg.model.m2det_config)
    init_net(net, cfg, args.trained_model)
    print_info('===> Finished constructing and loading model', ['yellow', 'bold'])
    net.eval()
    net = net.to(device)

    detector = Detect(cfg.model.m2det_config.num_classes,
                      cfg.loss.bkg_label, anchor_config, device=device)

    save_folder = os.path.join(cfg.outputs, args.dataset)
    _preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
    predictImage(save_folder=save_folder,
             net=net,
             detector=detector,
             device=device,
             transform=_preprocess,
             topk=100,
             iou_thresh=0.05,
             conf_threshod=0.5)

# https://github.com/KeepTryingTo/DeepLearning.git
