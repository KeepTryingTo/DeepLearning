"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/24-14:40
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

from scipy import linalg
import numpy as np


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    #TODO np.atleast_1d 用于确保输入的数组至少是一个 1-D（即一维）数组。
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'
    #TODO 计算两个均值向量之间的差异diff
    diff = mu1 - mu2
    '''
    print('&'*20)
    print(sigma1)#, sigma1.type())
    print('&'*20)
    print(sigma2)#, sigma2.type())
    '''
    # TODO 计算协方差 Product might be almost singular，
    #  使用linalg.sqrtm计算sigma1和sigma2乘积的平方根矩阵covmean。
    #  如果乘积矩阵接近奇异（即不可逆），则添加一个小的常数eps到两个协方差矩阵的对角线上，以避免数值问题。
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # TODO Numerical error might give slight imaginary component
    #  如果covmean包含复数部分（由于数值误差），则检查其对角线上的虚部是否接近于0。
    #  如果不是，抛出异常。如果虚部足够小，则只保留实部
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    #TODO 计算FID的公式为：||mu1 - mu2||^2 + Tr(sigma1) + Tr(sigma2) - 2*Tr(sqrt(sigma1*sigma2))，
    # 其中||mu1 - mu2||^2是均值向量差异的平方和，Tr表示矩阵的迹（对角线元素之和）。返回值是计算得到的FID值。

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

