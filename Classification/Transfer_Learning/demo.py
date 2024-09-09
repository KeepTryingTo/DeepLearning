"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/7/28-20:37
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from dataset import loadDataset
from matplotlib import cm as CM

trainLoader,testLoader = loadDataset()

print('trainLoader dataset size: {}'.format(len(trainLoader)))
print('testLoader dataset size: {}'.format(len(testLoader)))

iter_data = iter(trainLoader)
data = next(iter_data)
imgs,labels = data
print('imgs.shape: {}'.format(imgs.shape))
print('labels: {}'.format(labels))

def plot(imgs):
    # plt.figure(figsize=(8,6))
    fig,axs = plt.subplots(nrows=2,ncols=2)
    fig.suptitle("MNIST VISUALIZE")

    axs[0][0].imshow(np.asarray(imgs[0].squeeze()), cmap=CM.jet)
    axs[0][0].axis('off')
    axs[0][0].set_title(f'img one')

    axs[0][1].imshow(np.asarray(imgs[1].squeeze()), cmap=CM.jet)
    axs[0][1].axis('off')
    axs[0][1].set_title(f'img two')

    axs[1][0].imshow(np.asarray(imgs[2].squeeze()), cmap=CM.jet)
    axs[1][0].axis('off')
    axs[1][0].set_title(f'img three')

    axs[1][1].imshow(np.asarray(imgs[3].squeeze()), cmap=CM.jet)
    axs[1][1].axis('off')
    axs[1][1].set_title(f'img four')

    # plt.imshow()
    plt.show()

if __name__ == '__main__':
    plot(imgs)
    pass
