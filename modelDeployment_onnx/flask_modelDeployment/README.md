# -
使用VGG16网络模型训练数据和微调ResNet50训练数据进行图像识别<br>
pip install tensorflow==2.6.0<br>
pip install requirements.txt<br>
<hr>
train-VGG16:
      python main.py<br>
<hr>
train-ResNet50:
      python mainResNet50.py<br>
<hr>
predict:
      cd Flask<br>
      python Flask.py<br>
<hr>
      
<h3>（1）VGG16网络结构：</h3><br>
<a href='https://mydreamambitious.blog.csdn.net/article/details/123943751'>https://mydreamambitious.blog.csdn.net/article/details/123943751</a><br>
<hr>

<h3>（2） VGG16特征提取：</h3><br>
<a href='https://mydreamambitious.blog.csdn.net/article/details/123943372'>https://mydreamambitious.blog.csdn.net/article/details/123943372</a><br>
<hr>

<h3>（3）有关微调和迁移学习：</h3><br>
<a href='https://mydreamambitious.blog.csdn.net/article/details/123906833'>https://mydreamambitious.blog.csdn.net/article/details/123906833</a>
<br>
<hr>
<h3>This is about prediction's four images</h3><br>
<h3>VGG16模型预测结果</h3><br>
![Image text](https://github.com/KeepTryingTo/-/blob/main/images/vgg16_1.png)<br>
<img src='https://github.com/KeepTryingTo/-/blob/main/images/vgg16_1.png'><br>
![Image text](https://github.com/KeepTryingTo/-/blob/main/images/vgg16_2.png)<br>
<img src='https://github.com/KeepTryingTo/-/blob/main/images/vgg16_2.png'><br>
<hr>
<h3>微调ResNet50模型预测结果</h3><br>
![Image text](https://github.com/KeepTryingTo/-/blob/main/images/resnet50_1.png)<br>
<img src='https://github.com/KeepTryingTo/-/blob/main/images/resnet50_1.png><br>
![Image text](https://github.com/KeepTryingTo/-/blob/main/images/resnet50_2.png)<br>
<img src='https://github.com/KeepTryingTo/-/blob/main/images/resnet50_2.png'><br>

