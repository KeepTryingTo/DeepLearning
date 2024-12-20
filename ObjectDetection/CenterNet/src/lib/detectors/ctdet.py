from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
  # from src.lib.external.nms import soft_nms
  from torchvision.ops import nms
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from src.lib.models.decode import ctdet_decode
from src.lib.models.utils import flip_tensor
from src.lib.utils.image import get_affine_transform
from src.lib.utils.post_process import ctdet_post_process
from src.lib.utils.debugger import Debugger

from .base_detector import BaseDetector

class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    #TODO 输出检测结果
    with torch.no_grad():
      output = self.model(images)[-1]
      #TODO 得到heatmap以及预测高宽和中心偏移量
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      reg = output['reg'] if self.opt.reg_offset else None
      #TODO 图像输入到模型之前是否进行了翻转
      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      # torch.cuda.synchronize()
      forward_time = time.time()
      #TODO 预测结果的解码（boxes=[xmin,ymin,xmax,ymax]，预测分数，类别）
      dets = ctdet_decode(hm, wh, reg=reg,
                          cat_spec_wh=self.opt.cat_spec_wh,
                          K=self.opt.K)
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    #TODO 将预测的boxes映射回原图大小，注意返回的dets已经按照类别来进行划分保存了
    dets = ctdet_post_process(
        dets=dets.copy(),
        c=[meta['c']],
        s=[meta['s']],
        h=meta['out_height'],
        w=meta['out_width'],
        num_classes=self.opt.num_classes
    )
    #TODO 遍历类别，将其boxes缩放到指定的范围，默认batch size = 1，所以直接以dets[0]作为
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    #TODO 遍历每一个类别
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [
          detection[j] for detection in detections
        ], axis=0
      ).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         # soft_nms(results[j], Nt=0.5, method=2)
         pass
    scores = np.hstack(
      [
        results[j][:, 4] for j in range(1, self.num_classes + 1)
      ]
    )
    #TODO 判断预测的结果是否大于指定的每张图像可以预测最大数量
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      #TODO np.partition 是 NumPy 库中的一个函数，它用于对数组进行部分排序。其中kth为分区，
      # 整数或整数数组，指定要分区的位置。如果为整数，则数组被分区为两部分，使得 kth 位置左侧的
      # 元素都不大于 kth 位置右侧的元素。如果为整数数组，则每个元素指定了对应轴上的分区点。
      # 然后将左侧第kth个元素作为后面阈值划分的基准
      thresh = np.partition(scores, kth)[kth]
      #TODO 遍历每一个类别
      for j in range(1, self.num_classes + 1):
        #TODO 过滤掉那些低置信度的预测结果
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    #TODO 检测结果的缩放
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      #TODO 图像预处理
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      #TODO 获得预测的heatmap
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      #TODO
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results, img_name):
    debugger.add_img(image, img_id=img_name)
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        # print('bbox: {}'.format(bbox))
        if bbox[4] > self.opt.vis_thresh:
          # print('bbox: {}'.format(bbox))
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id=img_name)
    # debugger.show_all_imgs(pause=self.pause)
