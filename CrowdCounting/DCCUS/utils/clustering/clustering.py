import numpy as np

from PIL import ImageFile
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['Kmeans']


def preprocess_features(npdata, pca_dim=256, whitening=False, L2norm=False):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca_dim (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')
    #首先对数据的维度进行压缩之后得到更加低位的数据
    pca = PCA(pca_dim, whiten=whitening)
    npdata = pca.fit_transform(npdata)
    # L2 normalization
    if L2norm:
        row_sums = np.linalg.norm(npdata, axis=1)
        npdata = npdata / row_sums[:, np.newaxis]
    return npdata

class Clustering:
    def __init__(self, k, pca_dim=256, whitening=False, L2norm=False):
        self.k = k
        self.pca_dim = pca_dim
        self.whitening = whitening
        self.L2norm = L2norm
        
    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        # TODO 利用PCA将计算的特征均值和方差进行降维之后用于聚类 PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data, self.pca_dim, self.whitening, self.L2norm)
        # cluster the data 根据聚类算法对特征进行分类
        I = self.run_method(xb, self.k)
        self.images_lists = [[] for i in range(self.k)]
        #根据聚类之后的结果，对每一张图像进行划分集合
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)
        return None
    
    def run_method(self):
        print('Define each method')
    
class Kmeans(Clustering):
    def __init__(self, k, pca_dim=256, whitening=False, L2norm=False):
        super().__init__(k, pca_dim, whitening, L2norm)

    def run_method(self, x, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters)
        I = kmeans.fit_predict(x) #得到的是按照聚类的类别，对应每一张图像的标签，如0,...,K - 1
        return I
