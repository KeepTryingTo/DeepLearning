"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/3/1-9:56
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
#%%
'''
This script is for generating the ground truth density map 
for ShanghaiTech PartA. 
'''
import numpy as np
import scipy
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
from scipy.io import loadmat
import os
import glob
from matplotlib import pyplot as plt
import h5py
import PIL.Image as Image
from matplotlib import cm as CM
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


def generate_k_nearest_kernel_densitymap(image,points):
    '''
    Use k nearest kernel to construct the ground truth density map
    for ShanghaiTech PartA.
    image: the image with type numpy.ndarray and [height,width,channel].
    points: the points corresponding to heads with order [col,row].
    '''
    # the height and width of the image
    image_h = image.shape[0]
    image_w = image.shape[1]

    # coordinate of heads in the image
    points_coordinate = points
    # quantity of heads in the image
    points_quantity = len(points_coordinate)

    # generate ground truth density map
    densitymap = np.zeros((image_h, image_w))
    if points_quantity == 0:
        return densitymap
    else:
        pts = np.array(list(zip(np.nonzero(points_coordinate)[1],
                                np.nonzero(points_coordinate)[0])))  # np.nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）
        neighbors = NearestNeighbors(n_neighbors=4, algorithm='kd_tree',
                                     leaf_size=1200)  # https://blog.csdn.net/weixin_37804469/article/details/106911125
        neighbors.fit(pts.copy())
        # 计算当前的每一个标注位置到最近4个点的距离
        distances, _ = neighbors.kneighbors()

        for i, pt in enumerate(points_coordinate):
            pt2d = np.zeros((image_h,image_w), dtype=np.float32)
            if int(pt[1])<image_h and int(pt[0])<image_w:
                pt2d[int(pt[1]),int(pt[0])] = 1.
            else:
                continue
            if points_quantity > 3:
                sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            else:
                # sigma = np.average(np.array(points.shape))/2./2. #case: 1 point
                sigma = 15
            densitymap += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        return densitymap

def gaussian_filter_density_fixed(img, points):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.
    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.
    return:
    density: the density-map we want. Same shape as input image but only has one channel.
    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape=[img.shape[0],img.shape[1]]
    #print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    #print ('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        # sigma = 4 #np.average(np.array(gt.shape))/2./2. #case: 1 point
        # density += gaussian_filter(pt2d, sigma, truncate=7/sigma, mode='constant')
        sigma = 2
        density += gaussian_filter(pt2d, sigma, mode='constant')
    #print ('done.')
    return density

if __name__ == "__main__":
    root_dir = r'/home/ff/myProject/KGT/myProjects/myDataset/NWPU'
    save_dir = r'/home/ff/myProject/KGT/myProjects/myDataset/NWPU'

    ori_img_path = os.path.join(root_dir, 'images')
    ori_anno_path = os.path.join(root_dir, 'mats', 'mats')

    for phase in ['train', 'val']:
        sub_gt_save_dir = os.path.join(save_dir, phase, 'npys')
        sub_img_save_dir = os.path.join(save_dir, phase, 'imgs')
        if not os.path.exists(sub_gt_save_dir):
            os.makedirs(sub_gt_save_dir)
        if not os.path.exists(sub_img_save_dir):
            os.makedirs(sub_img_save_dir)
        with open(os.path.join(root_dir, 'mats', '{}.txt'.format(phase))) as f:
            lines = f.readlines()
            for i in lines:
                i = i.strip().split(' ')[0]
                im_path = os.path.join(ori_img_path, i + '.jpg')
                mat_path = os.path.join(ori_anno_path, i + '.mat')
                image = plt.imread(im_path)
                mat = loadmat(mat_path)['annPoints'].astype(np.float32)
                name = os.path.basename(im_path)
                im_save_path = os.path.join(sub_img_save_dir, name)
                # The Gaussian smoothed density map is just for visualization. It's not used in training.
                if (name.replace('.jpg','.npy') in os.listdir(sub_gt_save_dir)):
                    continue
                density_map = gaussian_filter_density_fixed(img=image, points=mat)
                dm_save_path = os.path.join(sub_gt_save_dir, name).replace('.jpg', '.npy')
                np.save(dm_save_path, density_map)

                print(name)

    for phase in ['test']:
        sub_save_dir = os.path.join(save_dir, phase, 'imgs')
        input_dataset_path_test = ori_img_path.replace('images', 'mats')
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        with open(os.path.join(input_dataset_path_test, '{}.txt'.format(phase))) as f:
            lines = f.readlines()
            for i in lines:
                i = i.strip().split(' ')[0]
                im_path = os.path.join(ori_img_path, i + '.jpg')
                mat_path = os.path.join(ori_anno_path, i + '.mat')
                image = plt.imread(im_path)
                mat = loadmat(mat_path)['annPoints'].astype(np.float32)
                name = os.path.basename(im_path)
                im_save_path = os.path.join(sub_img_save_dir, name)
                # The Gaussian smoothed density map is just for visualization. It's not used in training.
                if (name.replace('.jpg', '.npy') in os.listdir(sub_gt_save_dir)):
                    continue
                density_map = gaussian_filter_density_fixed(img=image, points=mat)
                dm_save_path = os.path.join(sub_gt_save_dir, name).replace('.jpg', '.npy')
                np.save(dm_save_path, density_map)

                print(name)


