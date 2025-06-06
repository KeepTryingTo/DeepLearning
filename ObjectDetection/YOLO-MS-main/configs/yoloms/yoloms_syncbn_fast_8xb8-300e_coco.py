# Reference to
# https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py
_base_ = 'mmyolo::rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py'

# ========================Frequently modified parameters======================
# -----data related-----
data_root = '/home/ff/myProject/KGT/myProjects/myDataset/coco/'
# Path of train annotation file
train_ann_file = 'captions/annotations/instances_train2014.json'
train_data_prefix = 'train2014/train2014/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'captions/annotations/instances_val2014.json'
val_data_prefix = 'val2014/val2014/'  # Prefix of val image path

# Number of classes for classification
num_classes = 80
# Batch size of a single GPU during training
train_batch_size_per_gpu = 8
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8
# persistent_workers must be False if num_workers is 0.
persistent_workers = True

# ========================Possible modified parameters========================
# -----data related-----
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 4
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 8

# Number of layer in MS-Block
layers_num = 3
# The scaling factor that controls the depth of the network structure
deepen_factor = 2 / 3
# The scaling factor that controls the width of the network structure
widen_factor = 0.8

# Input channels of PAFPN
in_channels = [320, 640, 1280]
# Middle channels of PAFPN
mid_channels = [160, 320, 640]
# Output channels of PAFPN
out_channels = 240

# Batch size of a single GPU during training
train_batch_size_per_gpu = 8

# Channel expand ratio for inputs of MS-Block
in_expand_ratio = 3
# Channel expand ratio for each branch in MS-Block
mid_expand_ratio = 2

# Channel down ratio for downsample conv layer in MS-Block
in_down_ratio = 2
# Normalization config
norm_cfg = dict(type='BN')
# Activation config
act_cfg = dict(type='SiLU', inplace=True)

# Kernel sizes of MS-Block in PAFPN
kernel_sizes = [1, (3, 3), (3, 3)]
strides = [8, 16, 32]

loss_bbox_weight = 2.0

# =======================Unmodified in most cases==================
model = dict(backbone=dict(_delete_=True,
                           type='YOLOMS',
                           arch='C3-K3579',
                           deepen_factor=deepen_factor,
                           widen_factor=widen_factor,
                           in_expand_ratio=in_expand_ratio,
                           mid_expand_ratio=mid_expand_ratio,
                           layers_num=layers_num,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg),
             neck=dict(_delete_=True,
                       type='YOLOMSPAFPN',
                       deepen_factor=deepen_factor,
                       widen_factor=widen_factor,
                       in_channels=in_channels,
                       mid_channels=mid_channels,
                       out_channels=out_channels,
                       in_expand_ratio=in_expand_ratio,
                       mid_expand_ratio=mid_expand_ratio,
                       layers_num=layers_num,
                       kernel_sizes=kernel_sizes,
                       in_down_ratio=in_down_ratio,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
             bbox_head=dict(
                 # type='RTMDetHead',
                 head_module=dict(
                     # type='RTMDetSepBNHeadModule',
                     widen_factor=widen_factor,
                     in_channels=out_channels,
                     feat_channels=out_channels,
                     num_classes=num_classes,
                     act_cfg=dict(
                         inplace=True,
                         type='LeakyReLU')
                      ),

                 # prior_generator=dict(
                 #     type='mmdet.MlvlPointGenerator',
                 #    offset=0, strides=strides
                 # ),
                 # bbox_coder=dict(type='DistancePointBBoxCoder'),

                 loss_bbox=dict(type='mmdet.DIoULoss',
                               loss_weight=loss_bbox_weight)
             ),
             train_cfg=dict(assigner=dict(num_classes=num_classes)))

# according to the label information of class_with_id.txt, set the class_name
# class_name = ('cat', )
# num_classes = len(class_name)
# metainfo = dict(
#     classes=class_name,
#     palette=[(220, 20, 60)]  # the color of drawing, free to set
# )

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(_delete_=True, type='yolov5_collate'),
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),
    dataset=dict(
        data_root=data_root,
        # metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        # metainfo=metainfo,
        data_prefix=dict(img=val_data_prefix),
        test_mode=True))

test_dataloader = val_dataloader

auto_scale_lr = dict(enable=True, base_batch_size=8 * 1)

# Reduce evaluation time
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')
test_evaluator = val_evaluator
