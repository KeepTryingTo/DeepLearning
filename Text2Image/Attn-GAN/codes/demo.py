"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/6/14-13:37
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
import sys
import os
import errno
import argparse
import numpy as np
from PIL import Image
from  codes.miscc.config import cfg
from torch.autograd import Variable
from nltk.tokenize import RegexpTokenizer
from codes.models.model import RNN_ENCODER,G_NET
from torchvision import transforms
from codes.dataset.datasets import TextDataset
from miscc.config import cfg, cfg_from_file

sys.path.insert(0, '../')
device = 'cuda'  # 'cpu' # 'cuda:0'


captions = ['the small bird is grey with a black crown and black bill.',
            'a smaller bird with an all grey body, a black nape, and a longer sharp bill.',
            'the bird has a black eyering and a black bill that is long.',
            'this bird has a grey body color with a few patches of brown on its crown and coverts']

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='./cfg/bird_attn2.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=0,help='number of workers(default: 4)')
    parser.add_argument('--stamp', type=str, default='normal',help='the stamp of model')
    parser.add_argument('--imsize', type=int, default=256,help='input imsize')
    parser.add_argument('--batch_size', type=int, default=4,help='batch size')
    parser.add_argument('--train', type=bool, default=True,help='if train model')
    parser.add_argument('--resume_epoch', type=int, default=20,help='resume epoch')
    parser.add_argument('--resume_model_path', type=str, default=r'../saved_models/bird/base_normal_bird_256_2024_06_20_13_00_13',help='the model for resume training')
    parser.add_argument('--multi_gpus', type=bool, default=False,help='if multi-gpu training under ddp')
    parser.add_argument('--gpu_id', type=int, default=0,help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true',default=True,
                        help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_single_imgs(imgs, save_dir, batch_size):
    for j in range(batch_size):
        folder = save_dir
        if not os.path.isdir(folder):
            #print('Make a new folder: ', folder)
            mkdir_p(folder)
        im = imgs[j].data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        filename = 'imgs_n%06d_gpu%1d.png'%(j, 1)
        fullpath = os.path.join(folder, filename)
        im.save(fullpath)
        print(f'save image {filename}...')


def rm_sort(caption, sorted_cap_idxs):
    non_sort_cap = torch.empty_like(caption)
    for idx, sort in enumerate(sorted_cap_idxs):
        non_sort_cap[sort] = caption[idx]
    return non_sort_cap

if __name__ == '__main__':
    args = parse_args()
    batch_size = 2
    # TODO 加载配置文件
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    noise = torch.randn(batch_size, cfg.GAN.Z_DIM).to(args.device)

    # TODO Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split='train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)

    # TODO 构建模型
    # TODO 文本编码模块，同时也冻结参数
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM).to(args.device)
    state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    for p in text_encoder.parameters():
        p.requires_grad = False
    print('Load text encoder from:', cfg.TRAIN.NET_E)
    text_encoder.eval()

    weight_path = r'/home/ff/myProject/KGT/myProjects/myProjects/AttnGAN-master/output/birds_attn2_2024_12_27_20_36_42/Model/netG_epoch_100.pth'
    netG = G_NET().to(args.device)
    netG.load_state_dict(torch.load(weight_path,map_location='cpu'))

    # TODO 对句子进行分词
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(captions)):
        tokens = tokenizer.tokenize(captions[i].lower())
        wordtoix = dataset.wordtoix
        sent_index = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if t in wordtoix:
                sent_index.append(
                    wordtoix[t]
                )
        #TODO 如果句子的长度超过指定的长度，则随机的从其中随着指定长度的词组成向量
        x = torch.zeros(size=(cfg.TEXT.WORDS_NUM, 1),dtype=torch.int32).to(args.device)
        sent_index = torch.tensor(sent_index,dtype=torch.int32)
        x_len = len(sent_index)
        num_words = x_len
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_index
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_index[ix[0]]
            x_len = cfg.TEXT.WORDS_NUM

        x = x.unsqueeze(dim=0)
        x_len = torch.tensor([x_len])
        sorted_cap_lens, sorted_cap_indices = torch.sort(x_len, 0, True)
        x = x[sorted_cap_indices].squeeze()
        x = x.unsqueeze(dim=0).repeat(batch_size,1).to(args.device)
        x_len = sorted_cap_lens.repeat(batch_size).to(args.device)

        # TODO 得到初始化的隐藏向量
        hidden = text_encoder.init_hidden(batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef TODO 对文本进行编码
        words_embs, sent_emb = text_encoder(x, x_len, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        # TODO 对于句子填充部分为0
        mask = (x == 0)
        # TODO 得到句子的词数量
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]

        #######################################################
        # TODO (2) Generate fake images
        ######################################################
        noise.data.normal_(0, 1)
        fake_imgs,_,_,_ = netG(noise, sent_emb,words_embs,mask)
        # save_single_imgs(imgs=fake_imgs, save_dir='../imgs', batch_size=batch_size, )
        import torchvision.utils as vutils

        if os.path.isdir(r'../imgs/gen_64_pixel') is False:
            os.mkdir(r'../imgs/gen_64_pixel')
        if os.path.isdir(r'../imgs/gen_128_pixel') is False:
            os.mkdir(r'../imgs/gen_128_pixel')
        if os.path.isdir(r'../imgs/gen_256_pixel') is False:
            os.mkdir(r'../imgs/gen_256_pixel')
        vutils.save_image(fake_imgs[0].data,f'../imgs/gen_64_pixel/{i}.png',nrow = batch_size,range=(-1,1),normalize = True)
        vutils.save_image(fake_imgs[1].data, f'../imgs/gen_128_pixel/{i}.png', nrow=batch_size, range=(-1, 1), normalize=True)
        vutils.save_image(fake_imgs[2].data, f'../imgs/gen_256_pixel/{i}.png', nrow=batch_size, range=(-1, 1), normalize=True)
    pass