import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_categories = [
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife",
        "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "TV", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

"""
reference to:
    pip install pycocotools-window
    https://cocodataset.org/#format-data
    https://blog.csdn.net/ayiya_oese/article/details/120994508
    https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
"""

#TODO  加载 COCO 数据集的注释文件
coco = COCO('voc_to_coco_format.json')

#TODO  假设 detections 是你模型的输出结果，包含 [image_id, bbox, score, class_id]
#TODO 结构：image_id: str(这个类型根据测试集中id类型来写即可), bbox: [xmin, ymin, width, height], score: float, class_id: int
detections = [
    # TODO 注意我们这里给出的image_id编号应该是在测试集中出现的，并且是符合测试集中编号格式
    #  不然报错 AssertionError: Results do not correspond to current coco set
    ["000001", [100, 100, 50, 50], 0.9, 1],
    ["000002", [110, 110, 50, 50], 0.75, 1],
]

def readClsFiles():
    # TODO 将 detections 转换为 COCO 格式
    results = []
    root = r'voc07_RefineDet'
    #TODO 遍历每一个检测结果类别文件
    for clsName in os.listdir(root):
        result_path = os.path.join(root,clsName)
        with open(result_path,'r',encoding='utf-8') as fp:
            lines = fp.readlines()
        for line in lines:
            det = line.strip().split(' ')
            image_id, score, class_id, xmin,ymin,xmax,ymax = det[0],det[1],det[2],det[3],det[4],det[5],det[6]
            # TODO COCO bbox 是 [xmin, ymin, xmax, ymax] => [xmin,ymin,w,h]
            bbox = [int(float(xmin)),int(float(ymin)),
                    int(float(xmax)) - int(float(xmin)),
                    int(float(ymax)) - int(float(ymin))]
            results.append({
                'image_id': str(image_id),
                'category_id': int(class_id) + 1, #TODO 注意类别索引设置是从1开始的
                'bbox': bbox,
                'score': float(score),
            })

    return results

def compute_map():
    results = readClsFiles()
    # TODO 将结果添加到 COCO，没有实际的 COCO 数据格式，因此使用 COCO的 "results" 格式
    coco_results = coco.loadRes(results)
    #TODO 注意这里的索引（类别索引）是从1开始的
    voc_categories = [i + 1 for i, cls_name in enumerate(VOC_CLASSES)]
    #TODO 自定义IOU的取值范围
    iou_thrs = [x / 100.0 for x in range(50, 96, 5)]  # 从 0.50 到 0.95

    # TODO 评估结果
    coco_eval = COCOeval(coco, coco_results, 'bbox')
    # coco_eval.params.catIds = voc_categories
    # coco_eval.params.iouThrs = iou_thrs
    # coco_eval.params.maxDets = [1, 10, 100, 200]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # TODO 获取每个类别的 AP（在类别数量与 coco_gt.getCatIds() 一一对应）
    precision_per_category = coco_eval.eval['precision']
    average_precisions = {}

    #TODO 在加载的 JSON 文件的 categories 字段中，包含了数据集中定义的所有类别。
    for i, cat_id in enumerate(coco.getCatIds()):
        ap = precision_per_category[0, :, i, 0, -1]
        average_precisions[cat_id] = ap.mean()

    #TODO 输出每个类别上的AP
    sum_ap = 0
    for cat_id, ap_value in average_precisions.items():
        print(f"Category ID {cat_id}: Average Precision = {ap_value:.3f}")
        sum_ap += ap_value
    print('@[0.5] mAP: {}'.format(sum_ap / len(coco.getCatIds())))
    # TODO 输出 mAP
    mAP = coco_eval.stats[0]  # mAP@IoU=0.50:0.95
    print(f'@[0.5:0.95] mAP: {mAP}')

if __name__ == '__main__':

    compute_map()
    pass