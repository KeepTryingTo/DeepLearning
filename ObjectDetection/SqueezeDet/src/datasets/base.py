import os

import numpy as np
import torch.utils.data

from src.utils.image import whiten, drift, flip, resize, crop_or_pad
from src.utils.boxes import compute_deltas, visualize_boxes


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, phase, cfg):
        super(BaseDataset, self).__init__()
        self.phase = phase
        self.cfg = cfg

    def __getitem__(self, index):
        #TODO 加载图像以及对应的id
        image, image_id = self.load_image(index)
        #TODO 获得对应图像的类别id以及box
        gt_class_ids, gt_boxes = self.load_annotations(index)

        image_meta = {'index': index,
                      'image_id': image_id,
                      'orig_size': np.array(image.shape, dtype=np.int32)}
        #TODO 对图像进行预处理，比如翻转，裁剪以及白化等操作
        image, image_meta, gt_boxes = self.preprocess(image, image_meta, gt_boxes)
        gt = self.prepare_annotations(gt_class_ids, gt_boxes)

        inp = {'image': image.transpose(2, 0, 1),
               'image_meta': image_meta,
               'gt': gt}

        #TODO 调试阶段使用到的，主要是判断上面的裁剪翻转等操作是否合理以及合法
        if self.cfg.debug == 1:
            image = image * image_meta['rgb_std'] + image_meta['rgb_mean']
            save_path = os.path.join(self.cfg.debug_dir, image_meta['image_id'] + '.png')
            visualize_boxes(image, gt_class_ids, gt_boxes,
                            class_names=self.class_names,
                            save_path=save_path)

        return inp

    def __len__(self):
        return len(self.sample_ids)

    def preprocess(self, image, image_meta, boxes=None):
        if boxes is not None:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0., image_meta['orig_size'][1] - 1.)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0., image_meta['orig_size'][0] - 1.)

        #TODO 偏移概率和翻转概率
        drift_prob = self.cfg.drift_prob if self.phase == 'train' else 0.
        flip_prob = self.cfg.flip_prob if self.phase == 'train' else 0.

        #TODO 对图像进行白化操作（白化（Whitening）是一种预处理技术，其主要作用是对数据进行变换，
        # 使数据的特征具有零均值和单位方差，从而消除特征之间的相关性。）
        image, image_meta = whiten(image, image_meta, mean=self.rgb_mean, std=self.rgb_std)
        image, image_meta, boxes = drift(image, image_meta, prob=drift_prob, boxes=boxes)
        image, image_meta, boxes = flip(image, image_meta, prob=flip_prob, boxes=boxes)
        #TODO 是裁剪填充还是直接缩放图像
        if self.cfg.forbid_resize:
            image, image_meta, boxes = crop_or_pad(image, image_meta, self.input_size, boxes=boxes)
            # print('crop image size...')
        else:
            image, image_meta, boxes = resize(image, image_meta, self.input_size, boxes=boxes)
            # print('resize image size...')

        return image, image_meta, boxes

    def prepare_annotations(self, class_ids, boxes):
        """
        :param class_ids:
        :param boxes: xyxy format
        :return: np.ndarray(#anchors, #classes + 9)
        """
        #TODO 计算gt box和anchorbox之间的iou，并获得对应的anchor box和gt box之间的中心偏移以及高宽比，
        # 还有就是最匹配的anchor box索引
        deltas, anchor_indices = compute_deltas(boxes, self.anchors)

        gt = np.zeros((self.num_anchors, self.num_classes + 9), dtype=np.float32)

        gt[anchor_indices, 0] = 1.  # mask
        gt[anchor_indices, 1:5] = boxes
        gt[anchor_indices, 5:9] = deltas
        gt[anchor_indices, 9 + class_ids] = 1.  # class logits

        return gt

    def get_sample_ids(self):
        raise NotImplementedError

    def load_image(self, index):
        raise NotImplementedError

    def load_annotations(self, index):
        raise NotImplementedError

    def save_results(self, results):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
