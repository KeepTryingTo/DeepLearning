'''
# -*- encoding: utf-8 -*-
# 文件    : viz_result.py
# 说明    : 保存结果
# 时间    : 2022/06/28 18:18:30
# 作者    : Hito
# 版本    : 1.0
# 环境    : TensorFlow2.3 or pytorch1.7
'''
import cv2
import os
import numpy as np



def viz_result(img_path,
               dets,
               dst_root):
    """
    :param dets: final bounding boxes: x1, y1, x2, y2, score
    @TODO: project(perspective) transformation back to frontal view
    @TODO: to facilitate recognition
    :param img_path:
    :param dets:
    :param dst_root:
    :return:
    """
    if not os.path.isfile(img_path):
        print('=> [Err]: invalid img file.')
        return
    img_name = os.path.split(img_path)[1][:-4]
    dst_orig_path = dst_root + '/' + img_name + '.jpg'
    dst_out_path = dst_root + '/' + img_name + '_out.jpg'
    
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # draw each bbox
    color = (0, 215, 255)
    for det in dets:
        # bbox corner and score
        pt_1 = (int(det[0] + 0.5), int(det[1] + 0.5))
        pt_2 = (int(det[2] + 0.5), int(det[3] + 0.5))
        score = str('%.3f' % (det[4]))

        print("dt: ", pt_1, ' ', pt_2, ' score:  ', score )
        # draw bbox
        cv2.rectangle(img=img, pt1=pt_1, pt2=pt_2, color=color, thickness=2)

        # compute score txt size
        txt_size = cv2.getTextSize(text=score,
                                   fontFace=cv2.FONT_HERSHEY_PLAIN,
                                   fontScale=2,
                                   thickness=2)[0]

        # draw text background rect
        pt_2 = pt_1[0] + txt_size[0] + 3, pt_1[1] - txt_size[1] - 5
        cv2.rectangle(img=img,
                      pt1=pt_1,
                      pt2=pt_2,
                      color=color,
                      thickness=-1)  # fill rectangle

        # draw text
        cv2.putText(img=img,
                    text=score,
                    org=(pt_1[0], pt_1[1]),  # pt_1[1] + txt_size[1] + 4
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2,
                    color=[225, 255, 255],
                    thickness=2)

        # draw landmarks
        if len(det) == 13:
            # format output landmark order?
            for pt_idx in range(4):
                cv2.circle(img=img,
                           center=(int(det[5 + pt_idx * 2]), int(det[5 + pt_idx * 2 + 1])),
                           radius=5,
                           color=(0, 0, 255),
                           thickness=-1)

            leftup = [det[5], det[6]]
            rightup = [det[7], det[8]]
            rightdown = [det[9], det[10]]
            leftdown = [det[11], det[12]]

            # perspective transformation
            src_pts = [leftup, rightup, rightdown, leftdown]
            out_img = perspective_transform(img=img, src_pts=src_pts)
            cv2.imwrite(dst_out_path, out_img)

    # write img to dst_root
    cv2.imwrite(dst_orig_path, img)
    # print('=> %s processed done.' % img_name)
    
    
def perspective_transform(img, src_pts):
    """
    perspective transform of 4 corner points of  license plate
    input 4 points
    :param img:
    :param src_pts:
    :return:
    """
    assert len(src_pts) == 4

    leftup = src_pts[0]
    rightup = src_pts[1]
    rightdown = src_pts[2]
    leftdown = src_pts[3]

    min_x = min(leftup[0], leftdown[0])
    max_x = max(rightup[0], rightdown[0])
    min_y = min(leftup[1], rightup[1])
    max_y = max(leftdown[1], rightdown[1])

    # construct 4 dst points
    dst_pts = []
    dst_leftup = [min_x, min_y]
    dst_rightup = [max_x, min_y]
    dst_rightdown = [max_x, max_y]
    dst_leftdown = [min_x, max_y]
    dst_pts = [dst_leftup, dst_rightup, dst_rightdown, dst_leftdown]

    src_pts, dst_pts = np.float32(src_pts), np.float32(dst_pts)
    persp_mat = cv2.getPerspectiveTransform(src=src_pts, dst=dst_pts)

    output = cv2.warpPerspective(src=img,
                                 M=persp_mat,
                                 dsize=(int(img.shape[1] * 1.5 + 0.5),
                                        int(img.shape[0] * 1.5 + 0.5)))
    return output