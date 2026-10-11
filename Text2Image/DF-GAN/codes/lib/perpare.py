

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from codes.lib.utils import mkdir_p, get_rank, load_model_weights
from codes.models.DAMSM import RNN_ENCODER, CNN_ENCODER
from codes.models.GAN import NetG, NetD, NetC

###########   preparation   ############
def prepare_models(args):
    device = args.device
    local_rank = args.local_rank
    n_words = args.vocab_size #TODO 词库大小
    multi_gpus = args.multi_gpus
    #TODO  图像编码器 （inception-v3）image encoder
    image_encoder = CNN_ENCODER(args.TEXT.EMBEDDING_DIM)
    img_encoder_path = args.TEXT.DAMSM_NAME.replace('text_encoder', 'image_encoder')
    state_dict = torch.load(img_encoder_path, map_location='cpu')
    image_encoder = load_model_weights(image_encoder, state_dict, multi_gpus=False)
    # TODO 冻结住 image_encoder.load_state_dict(state_dict)
    image_encoder.to(device)
    for p in image_encoder.parameters():
        p.requires_grad = False
    image_encoder.eval()
    # TODO 文本编码器 ，默认双向的LSTM作为文本编码器 text encoder
    text_encoder = RNN_ENCODER(n_words, nhidden=args.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(args.TEXT.DAMSM_NAME, map_location='cpu')
    text_encoder = load_model_weights(text_encoder, state_dict, multi_gpus=False)
    text_encoder.cuda()
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()
    #TODO  加载生成器，判别器以及 GAN models(generator and discriminator)
    netG = NetG(ngf=args.nf, nz=args.z_dim,
                cond_dim=args.cond_dim, imsize=args.imsize,
                ch_size=args.ch_size).to(device)
    netD = NetD(args.nf, args.imsize, args.ch_size).to(device)
    netC = NetC(args.nf, args.cond_dim).to(device)
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
    return image_encoder, text_encoder, netG, netD, netC


def prepare_dataset(args, split, transform):
    imsize = args.imsize
    if transform is not None:
        image_transform = transform
    elif args.CONFIG_NAME.find('CelebA') != -1:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
    else:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
    # train dataset
    from codes.lib.datasets import TextImgDataset as Dataset
    dataset = Dataset(split=split, transform=image_transform, args=args)
    return dataset


def prepare_datasets(args, transform):
    # train dataset
    train_dataset = prepare_dataset(args, split='train', transform=transform)
    # test dataset
    val_dataset = prepare_dataset(args, split='val', transform=transform)
    return train_dataset, val_dataset


def prepare_dataloaders(args, transform=None):
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_dataset, valid_dataset = prepare_datasets(args, transform)
    # train dataloader
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

