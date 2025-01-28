import cv2
import numpy as np
import pickle
import random
from collections import OrderedDict
import sys, os
import scipy.misc as misc
import torch

from PIL import Image

#-------------------------------------------------------------------------#
# dataloader for birds and flowers is modified from https://github.com/hanzhanggit/StackGAN
# don't set batch size 1
#-------------------------------------------------------------------------#

def resize_images(tensor, shape):
    if tensor.shape[1] == shape[1] and tensor.shape[0] == shape[0]:
        return tensor
    out = []
    for k in range(tensor.shape[0]):
        # tmp = misc.imresize(tensor[k], shape)
        tmp = cv2.resize(tensor[k],dsize=shape)
        out.append(tmp[np.newaxis,:,:,:])
    return np.concatenate(out, axis=0)

class Dataset(object):
    def __init__(self, workdir, img_size, batch_size, n_embed, mode='train'):
        if img_size in [256, 512]:
            self.image_filename = '304images.pickle'
            self.output_res = [64, 128, 256]
            if img_size == 512: self.output_res += [512]
        elif img_size in [64]:
            self.image_filename = '76images.pickle'
            self.output_res = [64]
        self.embedding_filename = 'char-CNN-RNN-embeddings.pickle'
        self.image_shape = [img_size, img_size, 3]

        self.batch_size = batch_size
        self.n_embed = n_embed

        self.imsize = min(img_size, 256)
        self.workdir = workdir
        self.train_mode = mode == 'train'
        print('wordir: {}'.format(workdir))
        self.get_data(os.path.join(self.workdir, mode))

        # set up sampler
        self._train_index = 0
        self._text_index = 0
        #TODO 图片的数量
        self._perm = np.arange(self._num_examples)
        np.random.shuffle(self._perm)
        self.saveIDs = np.arange(self._num_examples)

        print('>> Init basic data loader ', mode)
        print('\t {} samples (batch_size = {})'.format(self._num_examples, self.batch_size))
        print('\t {} output resolutions'.format(self.output_res))
        print ('\t {} embeddings used'.format(n_embed))
        
    def get_data(self, pickle_path):
        #TODO 加载图像数据集（图像数据集是存储在.pickle文件中）
        with open(os.path.join(pickle_path , self.image_filename), 'rb') as f:
            images = pickle.load(f)
            self.images = np.array(images)
        #TODO 加载embedding文件
        with open(os.path.join(pickle_path, self.embedding_filename), 'rb') as f:
            if sys.version_info.major > 2:
                embeddings = pickle.load(f,  encoding="bytes")
            else:
                embeddings = pickle.load(f)
            self.embeddings = np.array(embeddings)
            self.embedding_shape = [self.embeddings.shape[-1]]
            # print('embeddings: ', self.embeddings.shape)
        #TODO 加载图像文件名
        with open(os.path.join(pickle_path , 'filenames.pickle'), 'rb') as f:
            self.filenames = pickle.load(f)
            # print('list_filenames: ', len(self.filenames))
        #TODO birds数据集包含了200个类别，加载类别文件
        with open(os.path.join(pickle_path , 'class_info.pickle'), 'rb') as f:
            if sys.version_info.major > 2:
                class_id = pickle.load(f, encoding="bytes")
            else:
                class_id = pickle.load(f)
            self.class_id = np.array(class_id)

        self._num_examples = len(self.images)
        
    def readCaptions(self, filenames, class_id):
        name = filenames
        if name.find('jpg/') != -1:  # TODO 对于flowers dataset
            class_name = 'class_{0:05d}/'.format(class_id)
            name = name.replace('jpg/', class_name)
        cap_path = '{}/text_c10/{}.txt'.format(self.workdir, name)
        
        with open(cap_path, "r") as f:
            captions = f.read().split('\n')
        captions = [cap for cap in captions if len(cap) > 0]
        return captions

    def transform(self, images):
        
        transformed_images = np.zeros([images.shape[0], self.imsize, self.imsize, 3])
        ori_size = images.shape[1]
        # if ori_size < self.imsize:
        #     ori_size = int(self.imsize * (304/256))
        #     images = resize_images(images, shape=[ori_size, ori_size])

        for i in range(images.shape[0]):
            #TODO 计算裁剪的起始坐标位置
            if self.train_mode:
                h1 = int( np.floor((ori_size - self.imsize) * np.random.random()) )
                w1 = int( np.floor((ori_size - self.imsize) * np.random.random()) )
            else:
                h1 = int(np.floor((ori_size - self.imsize) * 0.5))
                w1 = int(np.floor((ori_size - self.imsize) * 0.5))
            #TODO 将裁剪的图像进行随机的翻转
            cropped_image = images[i][w1: w1 + self.imsize, h1: h1 + self.imsize, :]
            if random.random() > 0.5:
                transformed_images[i] = np.fliplr(cropped_image)
            else:
                transformed_images[i] = cropped_image

        return transformed_images

    def sample_embeddings(self, embeddings, filenames, class_id, sample_num):
        if len(embeddings.shape) == 2 or embeddings.shape[1] == 1:
            return np.squeeze(embeddings)
        else:
            #TODO 根据图像索引得到的embedding向量
            batch_size, embedding_num, _ = embeddings.shape
            # Take every sample_num captions to compute the mean vector
            sampled_embeddings = []
            sampled_captions = []

            for i in range(batch_size):
                #TODO replace=False表示不可以取相同元素；随机的选择sample_num个embedding向量
                randix = np.random.choice(embedding_num, sample_num, replace=False)
                if sample_num == 1:
                    randix = int(randix)
                    #TODO 读取captions句子
                    captions = self.readCaptions(filenames[i],class_id[i])
                    #TODO 根据选择的captions索引得到该句子
                    sampled_captions.append(captions[randix])
                    #TODO 得到对应句子captions的embedding
                    sampled_embeddings.append(embeddings[i, randix, :])
                else:
                    e_sample = embeddings[i, randix, :]
                    e_mean = np.mean(e_sample, axis=0)
                    #TODO 如果对图像描述的句子选择的sample_num数量不为1，
                    # 则将这些句子的embedding求解平均值得到最终对图像的描述句子embedding
                    sampled_embeddings.append(e_mean)
            sampled_embeddings_array = np.array(sampled_embeddings)
            
            return np.squeeze(sampled_embeddings_array), sampled_captions

    def get_index(self):

        start = self._train_index
        self._train_index += self.batch_size
        
        if (self._train_index+self.batch_size) > self._num_examples:
            np.random.shuffle(self._perm)
            start = 0
        end = start + self.batch_size
        
        return start, end
        
    def __iter__(self):
        return self

    def __next__(self):
        """Return the next `batch_size` examples from this data set."""

        n_embed = self.n_embed 
        # TODO 得到当前得到的数据集起始索引和结束索引位置shuffle
        start, end = self.get_index()
        #TODO _perm得到的是图像索引序列，通过起始位置和结束位置得到加载的数据batch中所有图像索引
        current_ids = self._perm[start:end]
        #TODO 根据图像数量_num_examples，随机的加载size大小的数据，用于mismatch image
        fake_ids = np.random.randint(self._num_examples, size=self.batch_size)

        #TODO 判断当前得到的mismatch是否和truth image相冲突
        collision_flag = (self.class_id[current_ids] == self.class_id[fake_ids])
        #TODO 对于相冲突的fake index进行偏移
        fake_ids[collision_flag] = (fake_ids[collision_flag] +
                                    np.random.randint(low=100, high=200)
                                    ) % self._num_examples
        
        images_dict = OrderedDict()
        wrongs_dict = OrderedDict()
        #TODO 根据索引得到图像数据
        sampled_images = self.images[current_ids]
        #TODO 得到mismatch image
        sampled_wrong_images = self.images[fake_ids, :, :, :]
        sampled_images = sampled_images.astype(np.float32)
        sampled_wrong_images = sampled_wrong_images.astype(np.float32)
        #TODO 裁剪和随机翻转操作
        sampled_images = self.transform(sampled_images)
        sampled_wrong_images = self.transform(sampled_wrong_images)
        images_dict = {}
        wrongs_dict = {}
        #TODO output_res = [64,128,256]
        for size in self.output_res:
            #TODO 根据指定的大小对图像缩放并对其进行归一化操作
            tmp = resize_images(sampled_images, shape=[size, size]).transpose((0,3,1,2))
            tmp = tmp * (2. / 255) - 1.
            images_dict['output_{}'.format(size)] = tmp

            tmp = resize_images(sampled_wrong_images, shape=[size, size]).transpose((0,3,1,2))
            tmp = tmp * (2. / 255) - 1.
            wrongs_dict['output_{}'.format(size)] = tmp

        ret_list = [images_dict, wrongs_dict]

        filenames = [self.filenames[i] for i in current_ids]
        class_id = [self.class_id[i] for i in current_ids]

        sampled_embeddings, sampled_captions = self.sample_embeddings(
            embeddings=self.embeddings[current_ids],
            filenames=filenames, class_id=class_id, sample_num=n_embed
        )
        ret_list.append(sampled_embeddings)
        ret_list.append(sampled_captions)

        ret_list.append(filenames)
        
        return ret_list

    def next_batch_test(self, max_captions=1):
        """TODO Return the next `batch_size` examples from this data set."""
        batch_size = self.batch_size
        
        start = self._text_index
        if (start + batch_size) > self._num_examples:
            end = self._num_examples
            self._text_index = 0
        else:
            end = start + batch_size
        self._text_index += batch_size

        sampled_images = self.images[start:end].astype(np.float32)
        sampled_images = self.transform(sampled_images)
        sampled_images = sampled_images * (2. / 255) - 1.
        
        sampled_embeddings = self.embeddings[start:end]
        _, embedding_num, _ = sampled_embeddings.shape
        sampled_embeddings_batchs = []
        
        sampled_captions = []
        sampled_filenames = self.filenames[start:end]
        sampled_class_id = self.class_id[start:end]
        for i in range(len(sampled_filenames)):
            captions = self.readCaptions(sampled_filenames[i],
                                         sampled_class_id[i])
            sampled_captions.append(captions)

        for i in range(np.minimum(max_captions, embedding_num)):
            batch = sampled_embeddings[:, i, :]
            sampled_embeddings_batchs.append(batch)

        return [sampled_images, sampled_embeddings_batchs, sampled_captions,
                self.saveIDs[start:end], self.class_id[start:end]]
