
import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from codes.lib.utils import choose_model


###########   preparation   ############
def load_clip(clip_info, device):
    import codes.clip as clip
    model = clip.load(clip_info['type'], device=device)[0]
    return model


def prepare_models(args):
    device = args.device
    local_rank = args.local_rank
    multi_gpus = args.multi_gpus
    #TODO 加载CLIP模型，都是采用ViT-B/32模型
    CLIP4trn = load_clip(args.clip4trn, device).eval()
    CLIP4evl = load_clip(args.clip4evl, device).eval()
    #TODO 加载生成器和判别器，以及CLIP的图像编码器和文本编码器
    NetG,NetD,NetC,CLIP_IMG_ENCODER,CLIP_TXT_ENCODER = choose_model(args.model)
    # image encoder TODO 冻结住CLIP图像编码器的参数
    CLIP_img_enc = CLIP_IMG_ENCODER(CLIP4trn).to(device)
    for p in CLIP_img_enc.parameters():
        p.requires_grad = False
    CLIP_img_enc.eval()
    # text encoder TODO 冻结住CLIP文本编码器的参数
    CLIP_txt_enc = CLIP_TXT_ENCODER(CLIP4trn).to(device)
    for p in CLIP_txt_enc.parameters():
        p.requires_grad = False
    CLIP_txt_enc.eval()
    # GAN models TODO 加载生成器和两个判别器
    netG = NetG(args.nf, args.z_dim, args.cond_dim, args.imsize, args.ch_size,
                args.mixed_precision, CLIP4trn).to(device)
    netD = NetD(args.nf, args.imsize, args.ch_size, args.mixed_precision).to(device)
    netC = NetC(args.nf, args.cond_dim, args.mixed_precision).to(device)
    #TODO 对于多GPU采用分布式训练
    if (args.multi_gpus) and (args.train):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG = torch.nn.parallel.DistributedDataParallel(netG, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
        netD = torch.nn.parallel.DistributedDataParallel(netD, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
        netC = torch.nn.parallel.DistributedDataParallel(netC, broadcast_buffers=False,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
    return CLIP4trn, CLIP4evl, CLIP_img_enc, CLIP_txt_enc, netG, netD, netC


def prepare_dataset(args, split, transform):
    #TODO 图像通道数以及数据增强
    if args.ch_size!=3:
        imsize = 256
    else:
        imsize = args.imsize
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip(),
            ])
    from codes.lib.datasets import TextImgDataset as Dataset
    dataset = Dataset(split=split, transform=image_transform, args=args)
    return dataset


def prepare_datasets(args, transform):
    # train dataset TODO 加载训练集和测试集
    train_dataset = prepare_dataset(args, split='train', transform=transform)
    # test dataset
    val_dataset = prepare_dataset(args, split='test', transform=transform)
    return train_dataset, val_dataset


def prepare_dataloaders(args, transform=None):
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_dataset, valid_dataset = prepare_datasets(args, transform)
    # train dataloader TODO 如果是多GPU的话就采用分布式训练
    if args.multi_gpus==True:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, sampler=train_sampler)
    else:
        train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, shuffle='True')
    # valid dataloader
    if args.multi_gpus==True:
        valid_sampler = DistributedSampler(valid_dataset)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, sampler=valid_sampler)
    else:
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, shuffle='True')
    return train_dataloader, valid_dataloader, \
            train_dataset, valid_dataset, train_sampler

