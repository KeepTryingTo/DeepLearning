"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/24-13:49
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
import torchvision.transforms as transforms
from PIL import Image

def mse_image_similarity(image1_path, image2_path):
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 可根据需要调整大小
        transforms.ToTensor(),  # 转换为 Tensor 格式
    ])

    img1_tensor = transform(img1)
    img2_tensor = transform(img2)

    mse_value = torch.mean((img1_tensor - img2_tensor) ** 2)

    return mse_value.item()  # 返回 MSE 数值

image1_path = 'path/to/image1.jpg'
image2_path = 'path/to/image2.jpg'
mse = mse_image_similarity(image1_path, image2_path)
print(f'Mean Squared Error (MSE): {mse}')
