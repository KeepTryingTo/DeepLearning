import torch
from torch.utils.data import Dataset, DataLoader
from utils.augmentations import SSDAugmentation
from data import *


def train_dataloader(dataset, image_size, batch_size, num_worksers):
    dataloader = get_dataloader(dataset,
                                batch_size=batch_size,
                                transform=SSDAugmentation(data_configs[dataset][str(image_size)]['min_dim'], MEANS),
                                image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                                num_worksers=num_worksers,
                                shuffle=True
                                )
    return dataloader


def get_dataloader(dataset, batch_size, transform, image_sets, num_workers, shuffle=True):
    if dataset == 'VOC':
        dataloader = DataLoader(
            VOCDetection(root=VOC_ROOT, image_sets=image_sets, transform=transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=detection_collate,
            pin_memory=True
        )

    return dataloader
