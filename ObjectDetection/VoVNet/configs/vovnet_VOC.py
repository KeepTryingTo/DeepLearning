model = dict(
    type='vovnet',
    input_size=300,
    init_net=True,
    rgb_means=(103.94, 116.78, 123.68),
    p=0.6,
    anchor_config=dict(
        feature_maps=[37, 18, 9, 5, 3, 2],
        steps=[8, 16, 32, 64, 100, 150],
        min_ratio=30,
        max_ratio=90,
        # aspect_ratios=[[1.6,2, 3], [1.6,2, 3], [1.6,2, 3],
        #                [1.6,2, 3], [1.6,2, 3],[1.6,2,3]],
        aspect_ratios=[[2, 3], [2, 3], [2, 3],
                       [2, 3], [2, 3],[2,3]],
        anchor_nums=[6, 6, 6, 6, 6, 6]
    ),
    num_classes=21,
    save_epochs=10,
    weights_save='weights/',
    pretrained_path = r'./weights/vovnet57_torchvision.pth'
)

train_cfg = dict(
    cuda=True,
    per_batch_size=32,
    lr=5e-3,
    # lr=0.00125,
    gamma=0.1,
    end_lr=5e-6,
    # end_lr=0.00000125,
    step_lr=[80000, 100000, 120000,160000],
    print_epochs=10,
    num_workers=8,
)

test_cfg = dict(
    cuda=True,
    topk=0,
    iou=0.005,
    soft_nms=True,
    score_threshold=0.75,
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
        train_sets=[('2014', 'train'), ('2014', 'valminusminival')],
        eval_sets=[('2014', 'minival')],
        test_sets=[('2015', 'test-dev')],
    )
)

import os
home = os.path.expanduser("~")
# VOCroot = os.path.join(home, "data/VOCdevkit/")
VOCroot = r'/home/ff/myProject/KGT/myProjects/myDataset/voc2012'
COCOroot = os.path.join(home, "data/coco/")
