# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")
#HOME = 'C:/Users/Sungjin Cho/Documents/'

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# PFP CONFIGS
'''
voc = {
    '300': {
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [40, 20, 10, 5],
        'min_dim': 320,
        'steps': [8, 16, 32, 64],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        'aspect_ratios': [[2], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'PFPNet_VOC_320'
    },
    '512': {
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [64, 32, 16, 8, 4, 2, 1],
        'min_dim': 512,
        'steps': [8, 16, 32, 64, 128, 256, 512],
        'min_sizes': [20, 51, 133, 215, 296, 378, 460],
        'max_sizes': [51, 133, 215, 296, 378, 460, 542],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'VOC_512',
    }
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
'''
# RefineDet CONFIGS
# TODO 512, coco edit
data_configs = {

    'VOC' : {
        '320': {
            'num_classes': 21,
            'img_size':320,
            'lr_steps': (80000, 100000, 120000),
            'max_iter': 1200000,
            'feature_maps': [40, 20, 10, 5],
            'min_dim': 320,
            'steps': [8, 16, 32, 64],
            'min_sizes': [32, 64, 128, 256],
            'max_sizes': [],
            'aspect_ratios': [[2], [2], [2], [2]],
            'variance': [0.1, 0.2],
            'clip': True,
            'name': 'RefineDet_VOC_320',
        },
        '512': {
            'num_classes': 21,
            'lr_steps': (80000, 100000, 120000),
            'max_iter': 120000,
            'feature_maps': [64, 32, 16, 8],
            'min_dim': 512,
            'steps': [8, 16, 32, 64],
            'min_sizes': [32, 64, 128, 256],
            'max_sizes': [],
            'aspect_ratios': [[2], [2], [2], [2]],
            'variance': [0.1, 0.2],
            'clip': True,
            'name': 'RefineDet_VOC_320',
        }
    }
    ,
    'COCO' : {
        'num_classes': 81,
        'lr_steps': (280000, 360000, 400000),
        'max_iter': 400000,
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [21, 45, 99, 153, 207, 261],
        'max_sizes': [45, 99, 153, 207, 261, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'COCO',
    }


}

