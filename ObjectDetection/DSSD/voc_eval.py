import torch
import numpy as np
import cv2
from tqdm import tqdm

from dataset.datasets.voc import BatchCollator
from configs.defaults import _C as cfg
from dataset.build_transforms import build_transforms

def sort_by_score(pred_boxes, pred_labels, pred_scores):
    #TODO 根据预测的confidence来进行从小到大的排序 => 排序之后的索引
    score_seq = [(-score).argsort()   for index, score in enumerate(pred_scores)]
    pred_boxes = [sample_boxes[mask]  for sample_boxes, mask in zip(pred_boxes, score_seq)]
    pred_labels = [sample_boxes[mask] for sample_boxes, mask in zip(pred_labels, score_seq)]
    pred_scores = [sample_boxes[mask] for sample_boxes, mask in zip(pred_scores, score_seq)]
    return pred_boxes, pred_labels, pred_scores

def iou_2d(cubes_a, cubes_b):
    """
    numpy 计算IoU
    :param cubes_a: [N,(x1,y1,x2,y2)]
    :param cubes_b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    """
    # 扩维
    cubes_a = np.expand_dims(cubes_a, axis=1)  # [N,1,4]
    cubes_b = np.expand_dims(cubes_b, axis=0)  # [1,M,4]

    # 分别计算高度和宽度的交集
    overlap = np.maximum(0.0,
                         np.minimum(cubes_a[..., 2:], cubes_b[..., 2:]) -
                         np.maximum(cubes_a[..., :2], cubes_b[..., :2]))  # [N,M,(w,h)]

    # 交集
    overlap = np.prod(overlap, axis=-1)  # [N,M]

    # 计算面积
    area_a = np.prod(cubes_a[..., 2:] - cubes_a[..., :2], axis=-1)
    area_b = np.prod(cubes_b[..., 2:] - cubes_b[..., :2], axis=-1)

    # 交并比
    iou = overlap / (area_a + area_b - overlap)
    return iou

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def eval_ap_2d(gt_boxes, gt_labels, pred_boxes,
               pred_labels, pred_scores,
               iou_thread, num_cls):
    """
    :param gt_boxes: list of 2d array,shape[(a,(x1,y1,x2,y2)),(b,(x1,y1,x2,y2))...]
    :param gt_labels: list of 1d array,shape[(a),(b)...],value is sparse label index
    :param pred_boxes: list of 2d array, shape[(m,(x1,y1,x2,y2)),(n,(x1,y1,x2,y2))...]
    :param pred_labels: list of 1d array,shape[(m),(n)...],value is sparse label index
    :param pred_scores: list of 1d array,shape[(m),(n)...]
    :param iou_thread: eg. 0.5
    :param num_cls: eg. 4, total number of class including background which is equal to 0
    :return: a dict containing average precision for each cls
    """
    all_ap = {}
    for label in range(num_cls)[1:]:
        # get samples with specific label 针对每一张图像，根据类别得到相应的gt box
        true_label_loc = [sample_labels == label for sample_labels in gt_labels]
        gt_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(gt_boxes, true_label_loc)]

        #针对每一张图像，根据类别标签得到对应的预测box和预测的confidence
        pred_label_loc = [sample_labels == label for sample_labels in pred_labels]
        bbox_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, pred_label_loc)]
        scores_single_cls = [sample_scores[mask] for sample_scores, mask in zip(pred_scores, pred_label_loc)]

        fp = np.zeros((0,))
        tp = np.zeros((0,))
        scores = np.zeros((0,))
        total_gts = 0
        # loop for each sample
        for sample_gts, sample_pred_box, sample_scores in zip(gt_single_cls, bbox_single_cls, scores_single_cls):
            total_gts = total_gts + len(sample_gts)
            assigned_gt = []  # one gt can only be assigned to one predicted bbox
            # loop for each predicted bbox
            for index in range(len(sample_pred_box)):
                scores = np.append(scores, sample_scores[index])
                if len(sample_gts) == 0:  # if no gts found for the predicted bbox, assign the bbox to fp
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
                    continue
                pred_box = np.expand_dims(sample_pred_box[index], axis=0)

                iou = iou_2d(sample_gts, pred_box)
                #得到预测框和gt box之间最大iou的预测框索引
                gt_for_box = np.argmax(iou, axis=0)
                #根据预测框索引，得到和gt box之间计算的iou最大值
                max_overlap = iou[gt_for_box, 0]
                if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt.append(gt_for_box)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
        # sort by score 根据confidence从大到小进行排序
        indices = np.argsort(-scores)
        fp = fp[indices]
        tp = tp[indices]
        # compute cumulative false positives and true positives
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # compute recall and precision
        recall = tp / total_gts
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = _compute_ap(recall, precision)
        all_ap[label] = ap
        # print(recall, precision)
    return all_ap

def createCfg(config_file = r'configs/resnet101_dssd320_voc0712.yaml'):
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

from dataset.myDir.transforms import Resize,SubtractMeans,ToTensor,Compose
transform = Compose(
    [
        Resize(cfg.INPUT.IMAGE_SIZE),
        SubtractMeans(cfg.INPUT.PIXEL_MEAN),
        ToTensor()
    ]
)
def begin(cfg,model,cls_file,
          eval_dataset,device):
    # TODO 读取类别文件
    num_classes = []
    with open(cls_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        num_classes.append(line.strip('\n').strip(''))

    print("INFO===>VOC CLASSES: {}".format(len(num_classes)))

    gt_boxes=[]
    gt_classes=[]
    pred_boxes=[]
    pred_classes=[]
    pred_scores=[]
    num=0

    cnt_false = 0
    cnt_true = 0
    for i in range(len(eval_dataset)):
        image,targets,index = eval_dataset.__getitem__(i)
        img_info = eval_dataset.get_img_info(i)

        boxes,classes = targets['boxes'],targets['labels']
        height, width = img_info['height'],img_info['width']

        image = image.unsqueeze(0)

        with torch.no_grad():
            detections,_,_=model(image.to(device))

            detection = detections[0]
            detection['boxes'][0::2] = detection['boxes'][0::2] / cfg.INPUT.IMAGE_SIZE
            detection['boxes'][1::2] = detection['boxes'][1::2] / cfg.INPUT.IMAGE_SIZE
            detection['boxes'][0::2] = detection['boxes'][0::2] * width
            detection['boxes'][1::2] = detection['boxes'][1::2] * height

            pred_boxes.append(detection['boxes'].cpu().numpy())
            pred_classes.append(detection['labels'].cpu().numpy())
            pred_scores.append(detection['scores'].cpu().numpy())

            # print('pred_boxes: ',detections['boxes'].size())
            # print('pred_classes: ',detections['labels'].size())
            # print('pred_scores: ',detections['scores'].size())

        gt_boxes.append(boxes)
        gt_classes.append(classes)
        num+=1
        print('[{}]============> [{}]'.format(eval_dataset.__len__(),num,end='\r'))

    print('cnt true: {}'.format(cnt_true))
    print('cnt false: {}'.format(cnt_false))

    # print(gt_boxes[0],gt_classes[0])
    # print(pred_boxes[0],pred_classes[0],pred_scores[0])
    #TODO 按照score对结果进行升序排序
    pred_boxes,pred_classes,pred_scores=sort_by_score(pred_boxes,pred_classes,pred_scores)
    all_AP=eval_ap_2d(
        gt_boxes=gt_boxes,
        gt_labels=gt_classes,
        pred_boxes=pred_boxes,
        pred_labels=pred_classes,
        pred_scores=pred_scores,
        iou_thread=0.005,
        num_cls=len(num_classes)+1)
    print("all classes AP=====>\n")

    mAP=0.
    for class_id,class_mAP in all_AP.items():
        mAP+=float(class_mAP)
        print('{} ===> {}'.format(num_classes[class_id - 1], class_mAP))
    mAP/=(len(num_classes)+1)
    print("mAP=====>%.3f\n"%mAP)

if __name__=="__main__":
    cfg = createCfg()
    from models.dssd_detector import DSSDDetector
    from dataset.datasets.voc import VOCDataset

    dir_root = r'/home/ff/myProject/KGT/myProjects/myDataset/voc2012/VOC2007'
    cls_file = r'configs/voc_classes.txt'
    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    val_transform = build_transforms(cfg, is_train=False)

    eval_dataset = VOCDataset(dir_root,
                              transform=val_transform,
                              split='test')

    print("INFO===>eval dataset has %d imgs" % len(eval_dataset))

    cfg = createCfg(config_file=r'configs/resnet101_dssd320_voc0712.yaml')
    model = DSSDDetector(cfg)
    # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print("INFO===>success convert BN to SyncBN")
    checkpoint = torch.load(r'./weights/voc_320_dssd.pth.tar', map_location='cpu')['model']
    model.load_state_dict(checkpoint)
    # model=convertSyncBNtoBN(model)
    # print("INFO===>success convert SyncBN to BN")
    model = model.to(device)
    model.eval()
    print("INFO===>success loading model")

    begin(cfg,model,cls_file,eval_dataset,device)

    """
    train: PASCAL VOC07 trainval + VOC12 trainval:
    test: PASCAL VOC07 test
    result: MAP = 0.643
    """
    pass

