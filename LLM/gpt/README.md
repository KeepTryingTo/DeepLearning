##基于transformer的对话模型实现
博文链接：[https://mp.weixin.qq.com/s/hha33dv5yISvlV_cFd5baA](https://mp.weixin.qq.com/s/hha33dv5yISvlV_cFd5baA "https://mp.weixin.qq.com/s/hha33dv5yISvlV_cFd5baA")

数据集下载地址：[https://modelscope.cn/datasets/qiaojiedongfeng/qiaojiedongfeng/files](https://modelscope.cn/datasets/qiaojiedongfeng/qiaojiedongfeng/files "https://modelscope.cn/datasets/qiaojiedongfeng/qiaojiedongfeng/files")

预训练模型下载地址：

链接：[https://pan.baidu.com/s/1CDVvsHll3M5uP6LZWGizsg](https://pan.baidu.com/s/1CDVvsHll3M5uP6LZWGizsg "https://pan.baidu.com/s/1CDVvsHll3M5uP6LZWGizsg") 
提取码：xkl4


数据集目录结构：
```

	划分训练集和测试集：python split_data.py
	构建词库:         python build_vocab.py 

	|---data/
		|---train.json
		|---val.json
		|---vocab.json


```


###模型训练
```
	python main.py
```

###模型测试
```
	python demo.py
```

