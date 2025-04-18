model = dict(
    type='Pelee',
    input_size=300,
    init_net=True,
    rgb_means=(103.94, 116.78, 123.68),
    p=0.6,
    anchor_config=dict(
        feature_maps=[37, 18, 9, 4, 2, 1],
        steps=[8, 16, 32, 64, 128, 300],
        min_ratio=20,
        max_ratio=90,
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        anchor_nums=[6, 6, 6, 6, 6, 6]
    ),
    num_classes=81,
    save_epochs=10,
    weights_save='weights/',
    pretained_model='weights/peleenet.pth'
)

train_cfg = dict(
    cuda=True,
    per_batch_size=8,
    lr=5e-4,
    gamma=0.1,
    end_lr=5e-6,
    step_lr=[200000, 300000, 400000, 500000],
    print_epochs=10,
    num_workers=8,
)

test_cfg = dict(
    cuda=True,
    topk=0,
    iou=0.45,
    soft_nms=True,
    score_threshold=0.01,
    keep_per_class=200,
    save_folder='eval',
)

loss = dict(overlap_thresh=0.5,
            prior_for_matching=True,
            bkg_label=0,
            neg_mining=True,
            neg_pos=3,
            neg_overlap=0.5,
            encode_target=False)

optimizer = dict(type='SGD', momentum=0.9, weight_decay=0.0005)

dataset = dict(
    VOC=dict(
        train_sets=[('2007', 'trainval'), ('2012', 'trainval')],
        eval_sets=[('2007', 'test')],
    ),
    COCO=dict(
        # train_sets=[('2014', 'train'), ('2014', 'valminusminival')],
        # eval_sets=[('2014', 'minival')],
        # test_sets=[('2015', 'test-dev')],
        train_sets=[('2014', 'train')],
        eval_sets=[('2014', 'val')]
    )
)

import os
home = os.path.expanduser("~")
VOCroot = os.path.join(home, "data/VOCdevkit/")
# COCOroot = os.path.join(home, "data/coco/")
COCOroot = r'/home/ff/myProject/KGT/myProjects/myDataset/coco'
