# PFPNet.pytorch
This repository provides the **official** PyTorch implementation for paper: [**Parallel Feature Pyramid Network for Object Detection**](https://openaccess.thecvf.com/content_ECCV_2018/html/Seung-Wook_Kim_Parallel_Feature_Pyramid_ECCV_2018_paper.html) by [**Seung-Wook Kim**](https://scholar.google.com/citations?hl=ko&user=UNZmEKIAAAAJ). 

**Note**: PFPNet is originally implemented on [Caffe](https://caffe.berkeleyvision.org). Following Caffe version, we re-implemented Pytorch version. 



![Architecture](./img/Architecture.png)



## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
  Note: We experiment on Pytorch 1.4

- Clone this repository.

- Then download the dataset by following the [instructions](#datasets) below.

- We now support [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) for real-time loss visualization and validation during training!

  


## Datasets

Currently, we only provide PFPNet of Pascal VOC version. 

### VOC Dataset

PASCAL VOC: Visual Object Classes

##### Download VOC2007 trainval & test

```
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval

```
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```



## Training

- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at: https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `PFPNet.pytorch/weights` dir:

```
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- Use the  following script below to train network .

```
python main.py --mode 'train' --dataset 'VOC' --save_folder 'weights/' --basenet './weights/vgg16_reducedfc.pth'
```

- Note:
  - For training, an NVIDIA GPU is strongly recommended for speed.
  - For instructions on Tensorboard usage/installation, see the [Installation](#installation) section.
  - You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `main.py` for options)

**Note**: COCO version and PFPNet512 are unavailable.

## Evaluation

To evaluate a trained network:

```
python main.py --mode 'test' --dataset 'VOC' --save_folder 'weights/' --test_model 'weights/PFPNetR320.pkl'
```

You can specify the parameters listed in the `main.py` file by flagging them or manually changing them. 




## Performance

VOC2007

mAP

| PFP320 | Paper version (Implemented by Caffe) | Pytorch version |
| ------ | ------------------------------------ | --------------- |
| mAP    | 80.7                                 | 80.7            |
| FPS    | 33                                   | 41              |

PFPNetR320: https://drive.google.com/file/d/1xEcdMGgmPNyopeNHEhTWQAbjFhl1LHAY/view?usp=sharing

## References

- [Original Implementation (CAFFE)](https://github.com/sfzhang15/RefineDet)
- A list of other great SSD ports that were sources of inspiration:
  - [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
  - [lzx1413/PytorchSSD](https://github.com/lzx1413/PytorchSSD)
  - [luuuyi/RefineDet.PyTorch](https://github.com/luuuyi/RefineDet.PyTorch)

