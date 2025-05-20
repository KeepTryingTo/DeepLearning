"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/5/19-21:13
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""


import os
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

root = r'images'
save_dir = r'./output'
img_size = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def save_images(image, mask, output_path, image_file, palette,num_classes):
    # Saves the image, the model output and the results after the post processing
    w, h = image.size
    image_file = os.path.basename(image_file).split('.')[0] # basenmae,返回图片名字
    colorized_mask = cam_mask(mask,palette,num_classes)
    colorized_mask.save(os.path.join(output_path, image_file+'.png'))
    return colorized_mask

def cam_mask(mask,palette,n):
    seg_img = np.zeros((np.shape(mask)[0], np.shape(mask)[1], 3))
    for c in range(n):
        seg_img[:, :, 0] += ((mask[:, :] == c) * (palette[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((mask[:, :] == c) * (palette[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((mask[:, :] == c) * (palette[c][2])).astype('uint8')
    colorized_mask = Image.fromarray(np.uint8(seg_img))
    return colorized_mask

#TODO 随机的生成RGB彩色
def create_color_map(num_classes):
    color_map = np.zeros((num_classes,3),dtype=np.uint8)
    for i in range(num_classes):
        color_map[i] = [np.random.randint(low=0,high=255) for _ in range(3)]
    return color_map

def output_to_color_image(output,color_map):
    color_image = np.zeros(shape = (output.shape[0],output.shape[1],3),dtype=np.uint8)
    for label in range(len(color_map)):
        color_image[output == label] = color_map[label]
    return color_image



palette = [
    (180, 120,  120),
    (6,   230,  230),
    (4,   200,  3),
    (204, 5,    255),
    (4,   250,  7),
    (235, 255,  7),
    (150, 5,    61),
    (120, 120,  70),
    (8,   255,  51),
    (255, 6,    82),
    (143, 255,  140),
    (204, 255,  4),
    (255, 6,    82),
    (204, 70,   3),
    (204, 70,   3),
    (0,   102,  200),
    (61,  230,  250),
    (11,  102,  255),
    (255, 7,    71),
    (9,   7,    230),
    (255, 122,  8)
]