# A Dual Refinement Mechanism for Real-World Visual Detection

## Inrtroduction
This repository is initially based on [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), and some modules are from [RFBNet](https://github.com/ruinmessi/RFBNet) and [deformable-convolution-pytorch](https://github.com/1zb/deformable-convolution-pytorch), a huge thank to them.

We currently only support **PyTorch-0.4.0** and **CUDA 8.0**. `./make.sh` is required for DeformableConv and COCO tools during installation.

## Models
Our models are aviliable at [GoogleDrive](https://drive.google.com/drive/folders/1_lgcOEOfdaiRCZrBIasYNGRjmH8YCFKF?usp=sharing) and [BaiduYun](https://pan.baidu.com/s/1I1FHecvQr8oVXe4kqvDxow)

## compiler deform_conv
```
#!/usr/bin/env bash
cd ./utils/

CUDA_PATH=/usr/local/cuda/

python build.py build_ext --inplace

cd ..

cd ./utils/deformconv/
nvcc -c -o deform_conv_cuda_kernel.cu.o deform_conv_cuda_kernel.cu -x cu -Xcompiler -fPIC -std=c++11
cd ..
CC=g++ python build_deformconv.py
cd ..
```

## Train
```
cd script
./train.sh
./train_trn.sh
```

## Eval
```
cd script
./batch_eval.sh
./batch_eval_trn.sh
```

## Paper

[arXiv paper](https://arxiv.org/abs/1807.08638) describes this project in detail.

**Xingyu Chen, Junzhi Yu, Shihan Kong, Zhengxing Wu, and Li Wen, "Towards Real-Time Accurate Object Detection in Both Images and Videos Based on Dual Refinement", *arXiv:1807.08638*, 2018.**
