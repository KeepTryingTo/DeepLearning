from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class MyDataSet:
    """自定义数据集"""

    def __init__(self,root_dir,transform = None):
        self.root_dir = root_dir
        self.transform = transform
    def ImageFold(self):
        datasets = ImageFolder(
            root=self.root_dir,transform=self.transform
        )
        return datasets


