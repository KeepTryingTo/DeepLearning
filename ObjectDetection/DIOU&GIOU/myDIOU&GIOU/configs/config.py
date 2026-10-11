num_classes = 21
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
mbox = [4, 6, 6, 6, 6, 4, 4]
variance = [0.1, 0.2]
feature_maps = [65, 33, 17, 9, 5, 3, 1] #TODO 输出的特征层的特征图大小

min_sizes = [  20.52,   51.3,   133.38,  215.46,  297.54,  379.62,  461.7 ]
max_sizes = [  51.3,   133.38,  215.46,  297.54,  379.62,  461.7,   543.78]

steps = [8, 16, 31, 57, 103, 171, 513] #TODO 表示每一层输出的下采样步长
top_k = 200

# detect settings
conf_thresh =  0.25
nms_thresh = 0.05

# Training settings
img_size = 513
batch_size = 4
epoch = 100
# lr_decay_epoch = 50
milestones = [120, 170, 220]

# data directory
root = '/home/ff/myProject/KGT/myProjects/myDataset/voc2012'

train_sets = [('2007','train'),('2007','val'),('2012', 'train'),('2012', 'val')]
test_sets = [('2007', 'test')]

means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
#TODO 由于上面我将batch size大小从24调整为4，因此，学习需要从0.001 => 0.000125
init_lr = 0.000125
weight_decay = 0.0005

VOC_CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

# dec evaluation
output_dir = 'output'

labelmap = ('aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
