[代码参考](https://github.com/CaptainEven/DenseBox.git)
> 对参考代码做了修改，将代码结构化，并使其可以正常训练，训练代码中没有进行测试。

# DenseBox
Baidu's Densebox implemention with PyTorch used for multi-task learning of object detection and landmark(key-point) localization

# Test result
![](https://github.com/CaptainEven/DenseBox/blob/master/demo_1.jpg)
![](https://github.com/CaptainEven/DenseBox/blob/master/demo_4.jpg) </br>
![](https://github.com/CaptainEven/DenseBox/blob/master/demo_2.jpg) </br>
![](https://github.com/CaptainEven/DenseBox/blob/master/demo_3.jpg) </br>

# Perspective transformation result
![](https://github.com/CaptainEven/DenseBox/blob/master/pair_1_1.jpg)
![](https://github.com/CaptainEven/DenseBox/blob/master/pair_1_2.jpg) </br>
 
![](https://github.com/CaptainEven/DenseBox/blob/master/pair_2_1.jpg)
![](https://github.com/CaptainEven/DenseBox/blob/master/pair_2_2.jpg) </br>

![](https://github.com/CaptainEven/DenseBox/blob/master/pair_3_1.jpg)
![](https://github.com/CaptainEven/DenseBox/blob/master/pair_3_2.jpg) </br>


## A small dataset sample: </br>
[patches](https://pan.baidu.com/s/1hGtdPHhuMW9Lz0gRLMYUUw) </br>

extract code: ir6n </br>


### 图片与标签格式
图片名字_label_[leftup_x]_[leftup_y]_[rightdown_x]_[rightdown_y].jpg
> eg: det_2018_09_13_000002_label_54_105_171_153.jpg

---
### 图片路径
训练图片均放在了dataset文件夹下面

---
### 测试图片
测试图片路径在 img 文件夹下
---
### 训练

```
python train.py
```
---
### 测试
```
python test.py
```
