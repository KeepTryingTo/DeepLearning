import torch
import cv2 as cv


def draw_box(img, labels, colors, names):
    if labels.shape[0] < 1:
        return img
    ret_img = img.copy()
    for weights, label, x1, y1, x2, y2 in labels:
        cv.rectangle(ret_img, (int(x1), int(y1)), (int(x2), int(y2)), color=colors[int(label)], thickness=2)
        cv.putText(ret_img, "{:s}".format(names[int(label)]), (int(x1), int(y1)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   colors[int(label)], 2)
        # cv.putText(ret_img, "{:.2f}".format(float(weights)), (int(x1), int(y1 + 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
        #            colors[int(label)],
        #            1)
    return ret_img



def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)






def bbox_areas(bboxes, keep_axis=False):
    x_min, y_min, x_max, y_max = (bboxes[:, 0],
                                  bboxes[:, 1],
                                  bboxes[:, 2],
                                  bboxes[:, 3])
    areas = (y_max - y_min + 1) * (x_max - x_min + 1)
    if keep_axis:
        return areas[:, None]
    return areas