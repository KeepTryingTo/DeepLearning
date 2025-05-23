import torch
import numpy as np
import cv2
import tqdm
from PIL import Image
from torchvision import transforms
from models.backbone import EfficientDetBackbone

def sort_by_score(pred_boxes, pred_labels, pred_scores):
    score_seq = [(-score).argsort() for index, score in enumerate(pred_scores)]
    pred_boxes = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, score_seq)]
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

def eval_ap_2d(gt_boxes, gt_labels,
               pred_boxes, pred_labels, pred_scores,
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
        # get samples with specific label
        true_label_loc = [sample_labels == label for sample_labels in gt_labels]
        gt_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(gt_boxes, true_label_loc)]

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
                gt_for_box = np.argmax(iou, axis=0)
                max_overlap = iou[gt_for_box, 0]
                if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt.append(gt_for_box)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
        # sort by score
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

if __name__=="__main__":
    from models.efficientdet.voc import VOCDataset,VOC_CLASS_LIST

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    params = Params(f'configs/voc.yml')

    from configs.config import *
    from models.efficientdet.augmentations import *
    from torch.utils.data import DataLoader
    eval_dataset = VOCDataset(
        root_dir=VOC_2007_PATH,
        transform=transforms.Compose(
            [
                Normalizer(mean=mean, std=std),
                Resizer(input_sizes[compound_coef])
            ]
        ),
        is_training=False
    )

    transform = transforms.Compose(
        [
            transforms.Resize(size=(512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std
            )
        ]
    )
    print("INFO===>eval dataset has %d imgs"%len(eval_dataset))
    eval_loader = DataLoader(eval_dataset,batch_size=1,
                                            shuffle=False,
                                            collate_fn=collater)

    model=loadModel(params,resum=r'./logs/VOC/efficientdet-d0_159_1000.pth')
    model=model.eval().to(device)
    print("===>success loading model")

    gt_boxes=[]
    gt_classes=[]
    pred_boxes=[]
    pred_classes=[]
    pred_scores=[]
    num=0

    from models.efficientdet.utils import BBoxTransform, ClipBoxes
    from utiles.utils import postprocess

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()


    loader = tqdm.tqdm(eval_loader)
    for i, data in enumerate(loader):
        img = data['img']
        boxes = data['annot'][0,:,:4]
        classes = data['annot'][0,:,4]

        image_id = eval_dataset.ids[i]
        image, width, height = eval_dataset._read_image(image_id)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(dim = 0)
        scale = np.array([width,height,width,height])

        with torch.no_grad():
            _, regression, classification, anchors=model(image.to(device))
            #TODO 进行后处理，过滤掉那些冗余的框
            out = postprocess(x=img,
                              anchors=anchors, regression=regression,
                              classification=classification,
                              regressBoxes=regressBoxes, clipBoxes=clipBoxes,
                              threshold=0.005, iou_threshold=0.45)

            #TODO [[xmin,ymin,xmax,ymax],...]
            reg_boxes = out[0]['rois']
            reg_boxes  = reg_boxes[...] / 512
            reg_boxes = reg_boxes * scale

            pred_boxes.append(reg_boxes)
            pred_classes.append(out[0]['class_ids'])
            pred_scores.append(out[0]['scores'])
        gt_boxes.append(boxes[0].numpy())
        gt_classes.append(classes[0].numpy())
        num+=1
        # print('===========================> {}'.format(num,end='\r'))

    # print(gt_boxes[0],gt_classes[0])
    # print(pred_boxes[0],pred_classes[0],pred_scores[0])

    pred_boxes,pred_classes,pred_scores=sort_by_score(pred_boxes,pred_classes,pred_scores)
    all_AP=eval_ap_2d(gt_boxes,gt_classes,pred_boxes,pred_classes,pred_scores,
                      0.5,len(VOC_CLASS_LIST))
    print("all classes AP=====>\n",all_AP)
    mAP=0.
    for class_id, class_mAP in all_AP.items():
        mAP+=float(class_mAP)
    mAP/=(len(VOC_CLASS_LIST))
    print("mAP=====>%.3f\n"%mAP)

