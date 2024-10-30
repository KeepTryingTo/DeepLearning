# config.py
import os.path

# gets home dir cross platform
home = os.path.expanduser("~")
ddir = os.path.join(home,"data/VOCdevkit/")

# note: if you used our download scripts, this should be right
VOCroot = r"/home/ff/myProject/KGT/myProjects/myDataset/voc2012" # path to VOCdevkit root dir
COCOroot = os.path.join(home,"data/COCO/")


#RFB CONFIGS
VOC_300 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1], #TODO 表示网络最后输出的特征图大小，分别用于检测小目标 -> 大目标
    'min_dim' : 300,#TODO 图像最小尺寸
    'steps' : [8, 16, 32, 64, 100, 300], #TODO 表示和上面特征图输出大小对应的下采样步长
    'min_sizes' : [30, 60, 111, 162, 213, 264],#TODO 可以发现最小和最大值之间是错开的
    'max_sizes' : [60, 111, 162, 213, 264, 315], #TODO

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]], #TODO anchor的高宽比，分别对应输出的6个特征图上的anchor高宽比
    'variance' : [0.1, 0.2], #TODO 用于gt box和anchor之间的比值时会用到
    'clip' : True, #TODO 是否进行裁剪
}

VOC_512= {
    'feature_maps' : [64, 32, 16, 8, 4, 2, 1],

    'min_dim' : 512,

    'steps' : [8, 16, 32, 64, 128, 256, 512],

    'min_sizes'  : [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8 ],

    'max_sizes'  : [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,
}


COCO_300 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [21, 45, 99, 153, 207, 261],

    'max_sizes' : [45, 99, 153, 207, 261, 315],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,
}

COCO_512= {
    'feature_maps' : [64, 32, 16, 8, 4, 2, 1],

    'min_dim' : 512,

    'steps' : [8, 16, 32, 64, 128, 256, 512],

    'min_sizes' : [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],

    'max_sizes' : [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,
}

COCO_mobile_300 = {
    'feature_maps' : [19, 10, 5, 3, 2, 1],

    'min_dim' : 300,

    'steps' : [16, 32, 64, 100, 150, 300],

    'min_sizes' : [45, 90, 135, 180, 225, 270],

    'max_sizes' : [90, 135, 180, 225, 270, 315],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,
}
