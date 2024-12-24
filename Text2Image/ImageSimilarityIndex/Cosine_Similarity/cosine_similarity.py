"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/24-14:12
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
import torchvision.transforms as transforms
from PIL import Image

def cosine_similarity_image(image1_path, image2_path):
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 可根据需要调整大小
        transforms.ToTensor(),  # 转换为 Tensor 格式
    ])

    img1_tensor = transform(img1).view(-1)  # 展平为一维向量
    img2_tensor = transform(img2).view(-1)  # 展平为一维向量

    cosine_similarity = torch.nn.functional.cosine_similarity(img1_tensor, img2_tensor, dim=0)

    return cosine_similarity.item()  # 返回余弦相似度数值

# 示例使用
image1_path = 'path/to/image1.jpg'
image2_path = 'path/to/image2.jpg'
cos_sim = cosine_similarity_image(image1_path, image2_path)
print(f'Cosine Similarity: {cos_sim}')