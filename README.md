<p align = "center">
	<a href = "https://blog.csdn.net/keep_trying_go/category_12736526.html"><img src = "https://img.shields.io/badge/Python-PyTorch-%23CC05FF"/></a>
	<a href = "https://blog.csdn.net/keep_trying_go/category_12736526.html"><img src = "https://img.shields.io/badge/PyTorch-Classification-%23CC05FF"/></a>
	<a href = "https://blog.csdn.net/keep_trying_go/category_12736526.html"><img src = "https://img.shields.io/badge/PyTorch-ObjectDetection-%23CC05FF"/></a>
	<a href = "https://blog.csdn.net/keep_trying_go/category_12736526.html"><img src = "https://img.shields.io/badge/PyTorch-CrowdCounting-%23CC05FF"/></a>
	<a href = "https://blog.csdn.net/keep_trying_go/category_12736526.html"><img src = "https://img.shields.io/badge/Android-modelDeployment-%23CC05FF"/></a>
</p>

<h2 align = "center"><a href = "https://blog.csdn.net/Keep_Trying_Go/article/details/140778634">深度学习CV</a>&<a href = "https://www.bilibili.com/video/BV1E2vMeTEkr/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">B站视频讲解</a></h2>

<p></p>
<h2>Projects</h2>
<a text-decoration="none" href = "https://blog.csdn.net/Keep_Trying_Go/article/details/140778634" >一.Classification</a><br/><p>主要使用PyTorch框架从0开始实现一个完整的图像分类过程，并且该过程可以作为一个图像分类的模版，后期直接在上面添加需要的额外功能；主要是实现MNIST和5种花的分类——<a href = "https://www.bilibili.com/video/BV1E2vMeTEkr/">视频讲解</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/141189496" >二.Pth2onnx</a><br/><p>将训练分类的模型转换为ONNX中间表示格式——<a href = "https://www.bilibili.com/video/BV1E2vMeTEkr?p=8">视频讲解</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/141264876" >三.ObjectDetection_pretrained_model</a><br/><p>加载PyTorch官方提供的预训练目标检测模型和将PyTorch官方提供的预训练目标检测模型转换为ONNX，最后实现图像的检测和实时检测——<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX?p=1">视频讲解</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/141355068" >四.CrowdCounting-FFNet</a><br/><p>论文发表于CVPR2024年最近的人群统计算法，提出该算法的目的在于当前的人群统计算法模型都比较复杂，并且还不够轻量化，因此FFNet这篇论文提出采用已有的轻量化并且和Vision Transformer有着同等特征提取能力的分类网络ConvNeXt-Tiny作为backbone主干网络，通过实验证明该模型相比于已有的人群统计模型更加的轻量化，并且实验效果最好;其中对FFNet的源码进行了详解——<a href = "https://www.bilibili.com/video/BV1N2WHesEPr?p=1">视频讲解</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/142053166" >五.Transfer_Learning</a><br/><p>该项目主要讨论了迁移学习相关的知识点，并且通过实验的方式我们在进行模型迁移的时候应该注意什么，以及我们什么时候该使用预训练模型做微调，实验过程中遇到的问题都例举了出来，对于学习迁移学习和微调具有很好的指引作用——<a href = "https://www.bilibili.com/video/BV1CGtNe7Esj/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/142208427" >六.classification & objectDetection modelDeployment_onnx</a><br/><p>该项目主要将图像分类模型和目标检测模型转换为ONNX中间表示格式，并且部署到Android系统中，实现了对单张图像的分类，物体检测以及实时的图像分类和物体检测，并且其中给出了视频的讲解，对于大家入门Android部署模型具有很好的学习价值——<a href = "https://www.bilibili.com/video/BV1jjsvejEu2?p=1">视频讲解</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/142523816" >七.classification & objectDetection modelDeployment_pt</a><br/><p>前面关于ONNX中间表示格式在Android中的部署，这里主要讲解将PyTorch训练的模型转化为torchscript中间表示格式，然后部署于Android系统，并且我这里讲解了在将PyTorch模型转换为torchscript中间表示格式时应该注意哪些事项，其中出现了哪些问题都是一一列举出来的，便于大家学习，最后在Android studio中加载我们的模型应该注意哪些都是在博文以及视频中讲解的——<a href = "https://www.bilibili.com/video/BV1jjsvejEu2?p=4">视频讲解</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/142708948" >八.segmentation modelDeployment</a><br/><p>这里主要是基于前面的Android部署模型的经验，将图像分割模型转换为ONNX部署到Android系统中，并且在博文和视频中我都讲解了如果将图像分割模型转换为torchscript中间表示格式在Android加载会存在很多问题，这些都是给出了问题的，并且最后在小结中还给大家一些建议关于Android中模型的部署——<a href = "https://www.bilibili.com/video/BV1jjsvejEu2?p=5">视频讲解</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/142730047" >九.Domain-General Crowd Counting in Unseen Scenarios（DCCUS）</a><p>主要对Domain-General Crowd Counting in Unseen Scenarios（DCCUS）论文进行了详解，这篇主要是人群计数领域单域泛化的提出，通过结合额外知识点，相关绘图以及代码对整个论文中所提到的算法实现都是进行了非常详细的讲解，讲解过程中所涉及的参考论文都是一一列举出来的，并且对于人群计数方向和单域泛化的的研究具有很好的价值——<a href = "https://www.bilibili.com/video/BV1N2WHesEPr?p=6">视频讲解</a></p>

<a text-decoration="none" href = "https://github.com/KeepTryingTo/DeepLearning/tree/main/modelDeployment/flask_modelDeployment_onnx" >十.flask modelDeployment</a><p>除了前面针对模型在Android系统中的部署，这里讲解了将模型使用flask框架进行部署，flask后端运行起来之后，，前端页面进行显示，比如点击单张图像，选择置信度阈值以及IOU阈值，选择检测的模型，最后点击检测；最后还实现了实时的检测，让功能看起来更加的完整——<a href = "https://www.bilibili.com/video/BV1jjsvejEu2?p=3">视频讲解</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/142588623" >十一.App_Android_CPP_NCNN</a><p>该项目主要是基于ncnn框架和Android studio对模型进行部署，首先将pytorch训练的模型转换torchscript中间表示格式，然后基于此利用pnnx工具将模型转换为ncnn支持的.nn.bin和.ncnn.param格式；其次对Android studio环境的配置（重点），这个已经在博文中列出了详细的步骤；最后是分类模型和目标检测模型的部署，部署过程涉及的核心点以及应该注意什么，我也在博文中详细讲解了——<a href = "">视频讲解</a></p>

<a text-decoration="none" href = "https://blog.csdn.net/Keep_Trying_Go/article/details/143133438" >论文Distribution Matching for Crowd Counting详解</a><p>其中对论文做了详细的讲解，同时对于其中涉及的原理以及方法，公式的推导都做了相关的讲解，最后对实验部分也做了详细的讲解<a href = "https://www.bilibili.com/video/BV1N2WHesEPr?vd_source=b2eaaddb2c69bf42517a2553af8444ab&p=7&spm_id_from=333.788.videopod.episodes">视频讲解</a></p>


<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143219789" >论文Distribution Matching for Crowd Counting算法代码讲解</a><p>这部分主要是论文主题算法的实现做了详细的讲解，公式的推导以及sinkhorn算法的迭代过程都做了比较详细的讲解<a href = "https://www.bilibili.com/video/BV1N2WHesEPr?vd_source=b2eaaddb2c69bf42517a2553af8444ab&p=7&spm_id_from=333.788.videopod.episodes">视频讲解</a></p>


<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143247917" >论文ASSD: Attentive Single Shot Multibox Detector详解（包含代码详解）</a><p>提出ASSD是建立在SSD算法的基础之上，ASSD算法提出的目的在于在特征空间构建相关特征。方法：对于全局相关信息，ASSD提出的方法更加强调在特征图中学习那些有用的区域，同时抑制那些相关的信息（注意力机制：让模型关注有用的区域），从而得到一个可靠的目标检测器。从设计方法上来讲，ASSD的方法设计相比于其他复杂的卷积神经网络的设计更加的简单和有效。
<a href = "">视频讲解</a></p>


