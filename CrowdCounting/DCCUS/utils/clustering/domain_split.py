import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.clustering import clustering
from scipy.optimize import linear_sum_assignment

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    #计算空间HW上的方差VAR和标准差STD
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C)
    #计算HW上的均值
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C)
    return feat_mean, feat_std
#TODO 开始的时候y_before为0，y_pred是根据聚类被分配的类别
"""
To avoid the large shift between previous and current sub-domain label 
assignments over clusters,we permute current assign
ments of sub-domain labels across clusters to achieve major agreement to
previous ones.We solve this as a Maximum Bipartite Matching problem using the Kuhn-Munkres algorithm(Munkres1957).
"""
def reassign(y_before, y_pred):
    assert y_before.size == y_pred.size
    #得到最大的类别数，也就是根据聚类分配的类别数
    D = max(y_before.max(), y_pred.max()) + 1
    #TODO 初始化一个全0的矩阵
    w = np.zeros((D, D), dtype=np.int64)
    #TODO 得到一个配对的矩阵，矩阵中每一个元素表示配对的数量，其实类似于一个混淆矩阵，表示多对应的类别
    for i in range(y_before.size):
        w[y_before[i], y_pred[i]] += 1
    #最大匹配算法
    # out = w.max() - w
    # print('out.size: {}'.format(out.shape))
    # print('out: {}'.format(out))
    #TODO 得到和聚类之后最大匹配的索引
    row_ind, col_ind= linear_sum_assignment(w.max() - w)
    return col_ind

def compute_features(dataloader, model, N, device):
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _, _) in enumerate(dataloader):
        with torch.no_grad():
            input_var = input_tensor.to(device)
            aux = model.domain_features(input_var).data.cpu().numpy()
            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')

            if i < len(dataloader) - 1:
                features[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = aux.astype('float32')
            else:
                # special treatment for final batch
                features[i * dataloader.batch_size:] = aux.astype('float32')
    return features

def compute_instance_stat(dataloader, model, N, device):
    model.eval()
    #返回图像文件名和对应的图像
    for i, (fname, input_tensor) in enumerate(dataloader):
        with torch.no_grad():
            input_var = input_tensor.to(device)
            #TODO 经过模型中间层的卷积结果
            #backbone -> sty_down -> unsqueeze(dim=0)
            conv_feats = model.conv_features(input_var)
            for j, feats in enumerate(conv_feats):
                #TODO 计算空间位置HW上的均值和标准差
                feat_mean, feat_std = calc_mean_std(feats)
                if j == 0:
                    aux = torch.cat((feat_mean, feat_std), 1).data.cpu().numpy()
                else:
                    aux = np.concatenate(
                        (aux, torch.cat((feat_mean, feat_std), 1).data.cpu().numpy()),
                        axis=1
                    )
            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')
            if i < len(dataloader) - 1:
                features[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = aux.astype('float32')
            else:
                # special treatment for final batch
                features[i * dataloader.batch_size:] = aux.astype('float32')
    print(features.shape)
    return features

def arrange_clustering(images_lists):
    #其中images_lists根据聚类算法划分类别之后的图像索引集合
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes) #得到排序之后图像索引
    #根据对图像的排序之后索引得到对应标签的索引
    return np.asarray(pseudolabels)[indexes] #得到图像根据聚类所对应标签
            

def domain_split(
        dataset, model, device, cluster_before,
        filename, epoch, nmb_cluster=3, method='Kmeans',
        pca_dim=256, batchsize=32, num_workers=32,
        whitening=False, L2norm=False, instance_stat=True
):
    #TODO 采用sklearn提供的聚类算法KMeans来实现对特征的聚类
    cluster_method = clustering.__dict__[method](nmb_cluster, pca_dim, whitening, L2norm)

    # dataset.set_transform('val')
    dataloader = DataLoader(
        dataset, batch_size=batchsize,
        shuffle=False, num_workers=0
    )
    #本文采用的计算方式为compute_instance_stat
    if instance_stat:
        #TODO 计算图像经过模型中间层的特征，然后计算均值和方差用于聚类
        features = compute_instance_stat(dataloader, model, len(dataset), device)
    else:
        features = compute_features(dataloader, model, len(dataset), device)
    #根据计算的特征再计算聚类结果
    clustering_loss = cluster_method.cluster(features, verbose=False)
    # print('images_lists: {}'.format(cluster_method.images_lists))
    #TODO 对聚类得到的结果的图像以及对应标签的顺序进行排序，得到排序之后的标签
    cluster_list = arrange_clustering(cluster_method.images_lists) #根据图像的索引以及聚类被分配的类别

    #TODO 聚类之前的标签和聚类之后的标签之间计算配对（采用最大匹配算法进行配对）
    mapping = reassign(cluster_before, cluster_list)
    #TODO 经过最大匹配KM算法之后的到的图像索引集合
    cluster_reassign = [cluster_method.images_lists[mapp] for mapp in mapping]
    # dataset.set_transform(dataset.split) 经过最大匹配KM算法之后，重新得到对图像分配的标签
    return arrange_clustering(cluster_reassign)

def demo():
    images_lists = [
            [1,5,7,9],[2,3,6,8],
            [12,13,14,11],[25,24,21,22]
        ]

    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)  # 得到排序之后图像索引
    # 根据对图像的排序之后索引得到对应标签的索引
    result = np.asarray(pseudolabels)[indexes]  # 得到图像根据聚类所对应标签
    return result

if __name__ == '__main__':
    demo()
    pass