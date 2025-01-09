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
from torch.autograd import Variable
from codes.lib.utils import load_models,merge_args_yaml
from nltk.tokenize import RegexpTokenizer

from codes.lib.perpare import prepare_dataloaders,prepare_models

sys.path.insert(0, '../')
device = 'cuda'  # 'cpu' # 'cuda:0'
batch_size = 4

captions = ['the small bird is grey with a black crown and black bill.',
            'a smaller bird with an all grey body, a black nape, and a longer sharp bill.',
            'the bird has a black eyering and a black bill that is long.',
            'this bird has a grey body color with a few patches of brown on its crown and coverts']

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='../cfg/bird.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=0,help='number of workers(default: 4)')
    parser.add_argument('--stamp', type=str, default='normal',help='the stamp of model')
    parser.add_argument('--imsize', type=int, default=256,help='input imsize')
    parser.add_argument('--batch_size', type=int, default=4,help='batch size')
    parser.add_argument('--train', type=bool, default=True,help='if train model')
    parser.add_argument('--resume_epoch', type=int, default=20,help='resume epoch')
    parser.add_argument('--resume_model_path', type=str,
                        default=r'/home/ff/myProject/KGT/myProjects/myProjects/DF-GAN-master/codes/saved_models/bird/base_normal_bird_256_2024_12_25_17_42_42',
                        help='the model for resume training')
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

def prepare_data(data, text_encoder):
    captions, caption_lens = data[0],data[1]
    captions, sorted_cap_lens, sorted_cap_idxs = sort_sents(captions, caption_lens)
    sent_emb, words_embs = encode_tokens(text_encoder, captions, sorted_cap_lens)
    sent_emb = rm_sort(sent_emb, sorted_cap_idxs)
    words_embs = rm_sort(words_embs, sorted_cap_idxs)
    return sent_emb, words_embs


def sort_sents(captions, caption_lens):
    #TODO  sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(input=caption_lens, dim=0, descending=True)
    captions = captions[sorted_cap_indices].squeeze()
    captions = Variable(captions).cuda()
    sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    return captions, sorted_cap_lens, sorted_cap_indices


def encode_tokens(text_encoder, caption, cap_lens):
    # encode text
    with torch.no_grad():
        if hasattr(text_encoder, 'module'):
            hidden = text_encoder.module.init_hidden(caption.size(0))
        else:
            hidden = text_encoder.init_hidden(caption.size(0))
        words_embs, sent_emb = text_encoder(caption, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    return sent_emb, words_embs


def rm_sort(caption, sorted_cap_idxs):
    non_sort_cap = torch.empty_like(caption)
    for idx, sort in enumerate(sorted_cap_idxs):
        non_sort_cap[sort] = caption[idx]
    return non_sort_cap

if __name__ == '__main__':
    args = merge_args_yaml(parse_args())
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    noise = torch.randn(batch_size, args.z_dim).to(args.device)

    train_dl, valid_dl, train_ds, valid_ds, sampler = prepare_dataloaders(args)
    args.vocab_size = train_ds.n_words
    # TODO 加载模型
    image_encoder, text_encoder, netG, netD, netC = prepare_models(args)
    image_encoder,text_encoder,netG,netD,netC = (image_encoder.to(args.device),text_encoder.to(args.device),
                                                 netG.to(args.device),netD.to(args.device),netC.to(args.device))
    weight_path = r'/home/ff/myProject/KGT/myProjects/myProjects/DF-GAN-master/codes/saved_models/bird/base_normal_bird_256_2024_12_25_17_42_42/state_epoch_115.pth'
    netG, netD, netC = load_models(netG,netD,netC,path=weight_path)
    # TODO 对句子进行分词
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(captions)):
        tokens = tokenizer.tokenize(captions[i].lower())
        wordtoix = train_ds.wordtoix
        sent_index = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if t in wordtoix:
                sent_index.append(
                    wordtoix[t]
                )
        #TODO 如果句子的长度超过指定的长度，则随机的从其中随着指定长度的词组成向量
        x = torch.zeros(size=(args.TEXT.WORDS_NUM, 1),dtype=torch.int32)
        sent_index = torch.tensor(sent_index,dtype=torch.int32)
        x_len = len(sent_index)
        num_words = x_len
        if num_words <= args.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_index
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:args.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_index[ix[0]]
            x_len = args.TEXT.WORDS_NUM
        data = [
            x.unsqueeze(dim=0).repeat(batch_size,1,1).to(args.device),
            torch.tensor(x_len).unsqueeze(dim=0).repeat(batch_size).to(args.device)
        ]
        sent_emb, word_emb = prepare_data(data, text_encoder)
        fake_imgs = netG(noise, sent_emb)
        # save_single_imgs(imgs=fake_imgs, save_dir='../imgs', batch_size=batch_size, )
        import torchvision.utils as vutils
        vutils.save_image(fake_imgs.data,
                          f'../imgs/{i}.png',
                          nrow = batch_size,range=(-1,1),
                          normalize = True)
    pass