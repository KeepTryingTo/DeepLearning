"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/24-14:08
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def psnr_image_similarity(image1_path, image2_path):
    # 加载图片
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 可根据需要调整大小
        transforms.ToTensor(),  # 转换为 Tensor 格式
    ])

    img1_tensor = transform(img1)
    img2_tensor = transform(img2)

    mse_value = torch.mean((img1_tensor - img2_tensor) ** 2).item()

    # TODO 防止除以零的情况
    if mse_value == 0:
        return float('inf')  # TODO PSNR 无穷大，表示完全相同

    max_pixel = 1.0  # 对于归一化到 [0, 1] 的情况
    psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(torch.tensor(mse_value)))

    return psnr_value.item()  # 返回 PSNR 数值

# 示例使用
image1_path = 'path/to/image1.jpg'
image2_path = 'path/to/image2.jpg'
psnr = psnr_image_similarity(image1_path, image2_path)
print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr} dB')