"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/24-14:41
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
import torchvision.transforms as transforms
from PIL import Image

def hamming_distance(image1_path, image2_path):
    img1 = Image.open(image1_path).convert('L')  # 转为灰度图
    img2 = Image.open(image2_path).convert('L')  # 转为灰度图

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 可根据需要调整大小
        transforms.ToTensor(),           # 转换为 Tensor 格式
    ])

    img1_tensor = transform(img1).view(-1)  # 展平为一维向量
    img2_tensor = transform(img2).view(-1)  # 展平为一维向量

    # TODO 将张量转换为二进制（0和1）
    img1_binary = (img1_tensor > 0.5).float()  # 假设阈值为0.5
    img2_binary = (img2_tensor > 0.5).float()  # 假设阈值为0.5

    hamming_dist = torch.sum(img1_binary != img2_binary).item()  # 计算不同位的数量

    return hamming_dist

image1_path = 'path/to/image1.png'
image2_path = 'path/to/image2.png'
distance = hamming_distance(image1_path, image2_path)
print(f'Hamming Distance: {distance}')