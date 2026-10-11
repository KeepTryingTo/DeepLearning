#encoding:utf-8
#
#created by xiongzihua
#
import os
import cv2
import torch
from PIL import Image
from configs.config import *
from utiles.nms import *
from configs import config
from collections import defaultdict
from tqdm import tqdm
from dataset.eval_voc_dataset import VOCDataSet

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np
VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]
#计算各个类别的AP
def voc_ap(rec,prec,use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0.,1.1,0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec>=t])
            ap = ap + p/11.

    else:
        # correct ap caculation
        mrec = np.concatenate(([0.],rec,[1.]))
        mpre = np.concatenate(([0.],prec,[0.]))
        for i in range(mpre.size -1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1],mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

#计算MAP
def voc_eval(
        preds,target,
        VOC_CLASSES=VOC_CLASSES,
        iou_threshold=0.05,use_07_metric=False
):
    '''
    preds:包含的是预测每一个类别的boxes，对应图像id，置信度以及boxes
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[x1,y1,x2,y2],]}
    '''
    aps = []
    #遍历VOC各个类别求解AP，然后求解最终的MAP
    for i,class_ in enumerate(VOC_CLASSES):
        #得到对应类别的预测值，也就是同一个类别的所有boxes框
        pred = preds[class_] #[[image_id,confidence,x1,y1,x2,y2],...]
        # 如果这个类别一个都没有检测到，那么赋值为-1
        if len(pred) == 0:
            ap = -1
            print('---class {} ap {}---'.format(class_,ap))
            aps += [ap]
            break
        #得到当前类别各个图像的image id
        image_ids = [x[0] for x in pred]
        #得到对应类别的所有boxes的置信度
        confidence = np.array([float(x[1]) for x in pred])
        #得到当前类别的所有预测的boxes
        BB = np.array([x[2:] for x in pred])
        # sort by confidence 将-confidence中的元素从小到大排列，提取其在排列前对应的index(索引)输出。
        #其实就是进行降序排序，并且sorted_ind是排序之前的索引号
        sorted_ind = np.argsort(-confidence)
        #-confidence降序排序，返回排序之后的值
        sorted_scores = np.sort(-confidence)
        #根据置信度的降序排列索引值得到对应的boxes框，也就是相当于对boxes进行了一个降序排列
        BB = BB[sorted_ind, :]
        #根据置信度的降序排列索引值，对image id进行降序排序
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.
        # target {(image_id,class):[[x1,y1,x2,y2],]}
        #其中key1,key2 == image_id,class
        for (key1,key2) in target:
            #从target中选择和当前遍历的类别相同的boxes框
            if key2 == class_:
                # 统计这个类别的正样本，在这里统计才不会遗漏
                #统计在target中当前类别对应的boxes数
                npos += len(target[(key1,key2)])
        #得到当前预测类别class对应的图像数
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        #遍历所用的图像，其中image_ids是预测结果中当前类别的图像id
        for d,image_id in enumerate(image_ids):
            #得到当前图像的预测bboxes
            bb = BB[d] #预测框
            #判断当前预测的图像id以及类别是否在target中
            if (image_id,class_) in target:
                #根据图像的id以及当前遍历的类别得到target中的bboxes
                BBGT = target[(image_id,class_)] #[[],]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    union = (bb[2]-bb[0]+1.)*(bb[3]-bb[1]+1.) + (bbgt[2]-bbgt[0]+1.)*(bbgt[3]-bbgt[1]+1.) - inters
                    #计算预测的box和target中的box之间的iou
                    if union == 0:
                        print(bb,bbgt)
                    
                    overlaps = inters/(union + 1e-6)
                    #大于给定的阈值，则对应的d索引位置为1
                    if overlaps > iou_threshold:
                        tp[d] = 1
                        # 这个框已经匹配到了，不能再匹配
                        BBGT.remove(bbgt)
                        if len(BBGT) == 0:
                            del target[(image_id,class_)] #删除没有box的键值
                        break
                fp[d] = 1-tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        #npos对应当前类别的正样本
        rec = tp/float(npos + 1e-6)
        """
        np.finfo使用方法
            eps是一个很小的非负数
            除法的分母不能为0的,不然会直接跳出显示错误。
            使用eps将可能出现的零用eps来替换，这样不会报错。
        """
        prec = tp/np.maximum(tp + fp, np.finfo(np.float64).eps)
        #print(rec,prec)
        ap = voc_ap(rec, prec, use_07_metric)
        print('---class {} ap {}---'.format(class_,ap))
        aps += [ap]
    print('---map {}---'.format(np.mean(aps)))

def predict_gpu(model,image_name,root_path=''):

    result = []
    image = cv2.imread(os.path.join(root_path,image_name))
    H,W,_ = image.shape
    img = cv2.resize(image,(448,448))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img)
    img = img.unsqueeze(dim = 0)
    img = img.to(device)

    pred = model(img) #1x7x7x30
    pred = pred.cpu()
    boxes,probs,cls_indexs,_ =  convert_cellboxes(
        predictions=pred,num_classes=VOC_NUM_CLASSES,conf_threshold=0.05
    )
    boxes,probs,cls_indexs = nms(boxes=boxes,probs=probs,cls_indexes=cls_indexs)

    for i,box in enumerate(boxes):
        # 得到经过NMS之后的框[cx,cy,w,h]
        xmin, ymin, xmax, ymax = box
        xmin, xmax, ymin, ymax = xmin.clamp(0, W), xmax.clamp(0, W), ymin.clamp(0, H), ymax.clamp(0, H)
        # 将其坐标还原回原图的大小
        xmin, ymin = int(xmin * W), int(ymin * H)
        xmax, ymax = int(xmax * W), int(ymax * H)
        xmin = np.minimum(xmin, W)
        ymin = np.minimum(ymin, H)
        xmax = np.minimum(xmax, W)
        ymax = np.minimum(ymax, H)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(xmin,ymin),(xmax,ymax),VOC_CLASSES[cls_index],image_name,prob])
    return result

def test_eval():
    preds = {'cat':[['image01',0.9,20,20,40,40],['image01',0.8,20,20,50,50],['image02',0.8,30,30,50,50]],'dog':[['image01',0.78,60,60,90,90]]}
    target = {('image01','cat'):[[20,20,41,41]],('image01','dog'):[[60,60,91,91]],('image02','cat'):[[30,30,51,51]]}
    voc_eval(preds,target,VOC_CLASSES=['cat','dog'])

if __name__ == '__main__':
    #TODO load the dataset
    voc_path = r'E:\conda_3\PyCharm\Transer_Learning\PASCAL_VOC'
    dataset = VOCDataSet(voc_root=voc_path,year='2007',train_set='test.txt')

    target =  defaultdict(list)
    preds = defaultdict(list)
    image_list = [] #image path list

    print('---prepare target---')
    size = dataset.__len__()
    for i in range(size):
        data = dataset[i]
        #这里的图像image_id其实也是图像的路径
        image_id = data['filename']
        image_list.append(data['filename'])
        for obj in data["object"]:
            # 将所有的gt box信息转换成相对值0-1之间
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(image_id))
                continue
            class_name = obj["name"]
            target[(image_id,class_name)].append([xmin,ymin,xmax,ymax])
    print('---start test---')
    from models.object.mySelfModel import EDANet
    from models.object.vgg_yolo import vgg16_bn
    from torchvision import models
    ######################################################################
    # model = EDANet(num_classes=VOC_NUM_CLASSES)
    ######################################################################
    # from models.object.resnet50 import YOLOv1ResNet
    # model = YOLOv1ResNet(
    #     B=B, S=S, C=VOC_NUM_CLASSES
    # )
    ######################################################################
    model = vgg16_bn()
    vgg = models.vgg16_bn(pretrained=True)
    new_state_dict = vgg.state_dict()
    dd = model.state_dict()
    for k in new_state_dict.keys():
        # print(k)
        if k in dd.keys() and k.startswith('features'):
            # print('yes')
            dd[k] = new_state_dict[k]
    model.load_state_dict(dd)
    ######################################################################
    checkpoint = torch.load(r'../weights/5.715_vgg_losses_t_best_model.pth.tar',map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    root = os.path.join(voc_path, "VOCdevkit", f"VOC{2007}")
    JPEG_PATH = os.path.join(root,"JPEGImages")
    for image_path in tqdm(image_list):
        result = predict_gpu(
            model,image_name=image_path,
            root_path= JPEG_PATH
        ) #result[[left_up,right_bottom,class_name,image_path],]
        # image_id is actually image_path image_id实际上是图像的路径
        for (x1,y1),(x2,y2),class_name,image_id,prob in result:
            preds[class_name].append([image_id,prob,x1,y1,x2,y2])
        # print(f'detect finish {image_path}......')
    
    print('---start evaluate---')
    voc_eval(preds,target,VOC_CLASSES=VOC_CLASSES,iou_threshold=0.05)