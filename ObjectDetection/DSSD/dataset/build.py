import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from dataset.build_transforms import build_transforms, build_target_transform
from structures.container import Container

from torch.utils.data import ConcatDataset

from dataset.path_catlog import DatasetCatalog
from dataset.datasets.voc import VOCDataset

_DATASETS = {
    'VOCDataset': VOCDataset
}


def build_dataset(dataset_list, transform=None,
                  target_transform=None, is_train=True):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data['args']
        #选择读取数据集的方式：VOC 或者COCO
        factory = _DATASETS[data['factory']]
        args['transform'] = transform
        args['target_transform'] = target_transform
        if factory == VOCDataset:
            args['keep_difficult'] = not is_train

        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)


class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])

        if self.is_train:
            list_targets = transposed_batch[1]
            targets = Container(
                {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
            )
        else:
            targets = None
        return images, targets, img_ids


def make_data_loader(cfg, is_train=True):
    #TODO 构建训练集的数据增强transforms
    train_transform = build_transforms(cfg, is_train=is_train)

    target_transform = build_target_transform(cfg) if is_train else None
    #TODO 根据当前指定的是训练集或者测试集，加载数据集
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    print('evaling dataset is ...... {}'.format(len(dataset_list)))
    datasets = build_dataset(
        dataset_list,
        transform=train_transform,
        target_transform=target_transform,
        is_train=is_train
    )

    shuffle = is_train

    data_loaders = []

    for dataset in datasets:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False
        )

        data_loader = DataLoader(
            dataset,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            collate_fn=BatchCollator(is_train)
        )
        data_loaders.append(data_loader)

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
