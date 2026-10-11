"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/6/14-13:37
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
import torchvision.utils as vutils

from codes.clip import clip
import sys
from tqdm import tqdm

sys.path.insert(0, '../')

from codes.lib.utils import load_model_weights, mkdir_p
from codes.models.RATLIP import NetG, CLIP_TXT_ENCODER


device = 'cpu'  # 'cpu' # 'cuda:0'
CLIP_text = "ViT-B/32"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.eval()

#TODO 文本编码器
text_encoder = CLIP_TXT_ENCODER(clip_model).to(device)
# NetG parms format the cfg TODO 生成网络模型
netG = NetG(64, 100, 512, 256, 3, False, clip_model).to(device)
path = r'D:\conda3\Transfer_Learning\GANs\RATLIP-main\codes\saved_models\state_epoch_050.pth'  # your path
checkpoint = torch.load(path, map_location=torch.device('cpu'))
netG = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus=False)

batch_size = 8
noise = torch.randn((batch_size, 100)).to(device)

captions = ['the small bird is grey with a black crown and black bill.\
a smaller bird with an all grey body, a black nape, and a longer sharp bill.\
the bird has a black eyering and a black bill that is long.\
this bird has a grey body color with a few patches of brown on its crown and coverts']

mkdir_p('./samples')
# generate from text
with torch.no_grad():
    for i in tqdm(range(len(captions))):
        caption = captions[i]
        #TODO 对描述场景的句子进行分词
        tokenized_text = clip.tokenize([caption]).to(device)
        #TODO 将句子token转换为句子嵌入向量和词嵌入向量
        sent_emb, word_emb = text_encoder(tokenized_text)
        #TODO 重复batch_size表示对于相同的场景描述生成多样性的图像
        sent_emb = sent_emb.repeat(batch_size, 1)
        #TODO 根据噪声和句子向量生成多样性的假图
        fake_imgs = netG(noise, sent_emb, eval=True).float()

        # name = f'{captions[i].replace(" ", "-")}'
        name = "bird"
        vutils.save_image(fake_imgs.data, './samples/%s.png' % (name),
                          nrow=8, value_range=(-1, 1), normalize=True)

# generate from text in sigle img
# with torch.no_grad():
#     batch_images = []
#     for i in range(len(captions)):
#         caption = captions[i]
#         tokenized_text = clip.tokenize([caption]).to(device)
#         sent_emb, word_emb = text_encoder(tokenized_text)
#         sent_emb = sent_emb.repeat(batch_size, 1)
#         fake_imgs = netG(noise, sent_emb, eval=True).float()

#         for j in range(batch_size):
#             fake_img = fake_imgs[j]
#             batch_images.append(fake_img)

#     # sigle
#     for idx, fake_img in enumerate(batch_images):
#         name = 'demo_%d' % (idx + 1)
#         vutils.save_image(fake_img, f'./samples/{name}.png',
#                           value_range=(-1, 1), normalize=True)