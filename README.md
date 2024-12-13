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

<a text-decoration="none" href = "https://blog.csdn.net/Keep_Trying_Go/article/details/143133438" >十二.论文Distribution Matching for Crowd Counting详解</a><p>其中对论文做了详细的讲解，同时对于其中涉及的原理以及方法，公式的推导都做了相关的讲解，最后对实验部分也做了详细的讲解<a href = "https://www.bilibili.com/video/BV1N2WHesEPr?vd_source=b2eaaddb2c69bf42517a2553af8444ab&p=7&spm_id_from=333.788.videopod.episodes">--视频讲解。</a></p>


<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143219789" >十三.论文Distribution Matching for Crowd Counting算法代码讲解</a><p>这部分主要是论文主题算法的实现做了详细的讲解，公式的推导以及sinkhorn算法的迭代过程都做了比较详细的讲解<a href = "https://www.bilibili.com/video/BV1N2WHesEPr?vd_source=b2eaaddb2c69bf42517a2553af8444ab&p=7&spm_id_from=333.788.videopod.episodes">--视频讲解。</a></p>


<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143247917" >十四.论文ASSD: Attentive Single Shot Multibox Detector详解（包含代码详解）</a><p>提出ASSD是建立在SSD算法的基础之上，ASSD算法提出的目的在于在特征空间构建相关特征。方法：对于全局相关信息，ASSD提出的方法更加强调在特征图中学习那些有用的区域，同时抑制那些相关的信息（注意力机制：让模型关注有用的区域），从而得到一个可靠的目标检测器。从设计方法上来讲，ASSD的方法设计相比于其他复杂的卷积神经网络的设计更加的简单和有效--
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143270390" >十五.论文Receptive Field Block Net for Accurate and Fast Object Detection详解（包含代码详解）</a><p>提出目的：由于当前的目标检测算法backbone部分依赖于像ResNet101或者其他特征提取模型，但是这些模型会导致整个推理流程变得很慢，实时性达不到。因此，要让目标检测模型能够应用到实际中，一个轻量化的模型是有必要的。但是问题在于实时性和准确率不可兼得，所以，怎么样去衡量速度和准确率是一个很重要的问题。提出方法：受到人类视觉系统感受野的结构启发，提出感受野模块（Receptive Fields Block，RFB），将感受野的偏心和大小考虑在内，从而提升特征的判别性和鲁棒性。最后，基于SSD算法，将RFB模块集成到其中并得到一个不错的效果。
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143313665" >十六.论文Feature-Fused SSD: Fast Detection for Small Objects 详解（包括代码详解）</a><p>提出目的：在有限的图像分辨率和包含有限信息的情况下，检测图像或者视频中的小目标是具有很大的挑战，但是大量的方法中都是以牺牲速度来提升精度。为了快速检测小目标，同时保持精度不下降是本文主要的目的。提出方法：基于SSD目标检测算法，提出了多层特征融合的方法，目的在于融合上下文信息。为了提升检测小目标的精度，本文设计了两个特征融合模块，其中融合操作包含拼接和逐元素求和，这两种方法对于最终的检测效果有所区别--<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "" >十七.论文FSSD: Feature Fusion Single Shot Multibox Detector详解（代码详解）</a><p>提出目的：作者认为最初的SSD算法特征金字塔检测算法不能很好的融合不同尺度的特征，因此FSSD提出了新的特征融合方式，从而提升原有的SSD性能。提出方法：FSSD基于SSD算法提出一种新的特征融合方法，并且速度上只有一点下降，相比于检测准确率的提升是值得的。具体的方法是：对不同层不同尺度的特征进行拼接，随后通过下采样得到新的特征金字塔，最后的输出层具有不同感受野大小，分别用于检测不同尺度的物体--<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143603094" >十八.ISSD算法详解</a><p>提出目的：虽然SSD算法已经很快了，但是在精度上和最好的精度相比还是具有一定的差距，因此针对这个问题特别提出了ISSD。提出方法：本文基于SSD算法的基础来进行改进，在不影响速度的情况下提高模型的精度，使用比较常用的Inception模块代替SSD中的部分模块，同时整体模型架构在不增加原有模型复杂度的同时，也增加了性能，其中在模块中采用了BN和残差结构（弥补信息的丢失，防止梯度消失）；最后还提升了NMS算法，克服了模型表达能力的缺陷--
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143461579" >十九.DSSD算法详解</a><p>提出目的：作者认为原始的SSD算法在head层包含的上下文信息不足，并且如果直接在深层（head层）添加卷积操作操作以达到增加上下文信息不是那么容易的，简单的设计很容易失败。提出方法：基于原始的SSD算法 + Residual101 + 反卷积（上采样）层将增加更多的上下文信息，同时提升算法检测物体的效果，特别是小目标的检测。具体方法是通过在backbone的输出层并且进行上采样之后逐元素相乘最后通过head层，每一层输出都是这样（类似对称的encoder-decoder）--
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143495883" >二十.GIOU算法详解</a><p>提出目的：目标检测算法中IOU常用于评估定位框的预测效果，但是直接使用IOU作为预测框回归距离的评估并最大化评估值存在一定的问题，在二维的预测坐标框回归中，IOU可以作为其回归损失优化，但是IOU面对预测框和真实框之间没有重叠的情况时会出现停止更新，因为真实框和预测框之间没有重叠就代表IOU为0，梯度无法更新。提出方法：本文为了解决预测框和真实框之间没有重叠的情况提出了GIOU，并且相比于IOU Loss实现了不错的效果，一定程度上缓解了非重叠框之间梯度为0的情况--
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143466301" >二十一.M2Det算法详解</a><p>提出目的：特征金字塔受到了很大的关注，并且在很多论文中都得到了应用，以此来避免引起的尺度变化问题。虽然之前的方法在引入特征金字塔时带来了性能的提升，但是他们仅仅只是简化并且是以固有的尺度去构建结构。提出方法：M2Det提出了更加有效的特征金字塔结构检测不同尺度的物体。具体方法是首先融合了来自backbone多层特征得到基准特征，其次是将基准输入到TUM（Thinned U-shape Modules）和FFM（Feature Fusion Modules）模块，并且最后将每一个TUM的解码层输出层相同尺度构建一个特征金字塔。
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143477078" >二十二.DIOU算法详解</a><p>提出目的：目标检测领域除了正确预测物体之外，其中对物体的正确定位也是非常重要的，在已经存在的目标检测方法中，对于坐标框的回归一般都是𝑙𝑛l_n损失，但是并没有用于验证评估，比如使用IOU作为预测框的评估。其中IOU Loss和GIOU Loss已经被提出用于IOU评估，但是任然存在收敛慢和不能正确定位问题。提出方法：本文提出DIOU（Distance-IOU）Loss作为预测框和真实框之间距离的评估，相比于IOU Loss和GIOU损失收敛更快，并且本文还总结了影响坐标框回归的三个因素，①重叠面积②中心点距离③预测 box和真实box的比率大小，基于此还提出了一个CIOU（Complete-IOU） Loss，而且收敛更快和最终的性能更好。
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://blog.csdn.net/Keep_Trying_Go/article/details/143724189?sharetype=blogdetail&sharerId=143724189&sharerefer=PC&sharesource=Keep_Trying_Go&spm=1011.2480.3001.8118" >二十三.论文Pelee: A Real-Time Object Detection System on Mobile Devices详解（代码详解）</a><p>提出目的：目前已经有大量轻量化模型被提出，比如MobileNet，shuffleNet以及MobileNetv2等模型，这些模型大部分都依赖深度可分离卷积，但是这个深度可分离卷积在深度学习框架并没有被有效的使用。提出方法：本文基于传统的卷积提出一个有效的网络架构PeleeNet， PeleeNet 网络模型具体包含Dense Layer，Stem Block以及Transition Layer层；并且基于PeleeNet 网络对SSD算法进行优化处理，包含特征图的选择，残差模块和小的卷积核，最后在PASCAL VOC上的测试结果为76.4%的MAP。
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX?vd_source=b2eaaddb2c69bf42517a2553af8444ab&spm_id_from=333.788.videopod.episodes&p=20">视频讲解。</a></p>

<a text-decoration="none" href = "https://blog.csdn.net/Keep_Trying_Go/article/details/143804612?sharetype=blogdetail&sharerId=143804612&sharerefer=PC&sharesource=Keep_Trying_Go&spm=1011.2480.3001.8118" >二十四.论文DSOD: Learning Deeply Supervised Object Detectors from Scratch详解（代码详解）</a><p>提出目的：当前最好的目标检测算法都是基于预训练模型来进行训练的，比如像SSD算法是基于预训练的VGG16来训练backbone的，尽管通过微调的方式一定程度缓解从头开始训练的问题，那如果没有这些在大规模数据集上训练的预训练模型该怎么办呢？并且这些预训练模型很可能和我们自己的任务大不相关。因此，这篇论问基于这个问题提出从头开始训练自己的目标检测模型。提出方法：基于上面出现的现象，作者提出DSOD从头开始训练自己的模型，由于之前复杂的损失函数和有限的训练数据集，导致训练过程很容易失败，因此文本提出从头开始训练模型的一系列方法，这篇文章提出的关键方法就是采用密集连接进行模型的深度监督训练（有监督训练）（基于SSD框架）。
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX?vd_source=b2eaaddb2c69bf42517a2553af8444ab&spm_id_from=333.788.videopod.episodes&p=22">视频讲解。</a></p>

<a text-decoration="none" href = "https://blog.csdn.net/Keep_Trying_Go/article/details/143664068?sharetype=blogdetail&sharerId=143664068&sharerefer=PC&sharesource=Keep_Trying_Go&spm=1011.2480.3001.8118" >二十五.目标检测数据集COCO的MAP评价指标工具pycocotools用于VOC或者自定义类别数据集的MAP计算以及类别的AP结果展现（代码详解）</a><p> 我们都知道目标检测领域有自己的评价指标MAP，表示计算每个类别上的AP（Average Precision）的结果，最后求取平均值得到MAP。但是像COCO大型的目标检测数据集有自己的MAP评价工具pycocotools，由于COCO是80个类别，如果要将这个工具应用到自己的领域，比如自定义了各个类别的数据集或者像20个类别VOC数据集，该怎么做呢？
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX?vd_source=b2eaaddb2c69bf42517a2553af8444ab&spm_id_from=333.788.videopod.episodes&p=15">视频讲解。</a></p>


<a text-decoration="none" href = "https://blog.csdn.net/Keep_Trying_Go/article/details/143666412?sharetype=blogdetail&sharerId=143666412&sharerefer=PC&sharesource=Keep_Trying_Go&spm=1011.2480.3001.8118" >二十六.论文Soft-NMS– Improving Object Detection With One Line of Code 详解</a><p>提出目的：目标检测算法中的后处理部分使用NMS对重叠的框进行过滤，对于相同类别的重叠部分的框，置信度最高的将被保留，和最高置信度IOU大于指定阈值的将被过滤掉，但是这样存在一个问题，就是确实存在两个相同类别的物体靠的很近，那么导致检测的框很大部分会重叠，那么使用NMS算法进行过滤将导致其中一个框被过滤掉，这并不是我们想要的。提出方法：本文提出Soft-NMS算法，对于和最高置信度IOU大于指定阈值的那些框，对其置信度进行一定的微调，然后再过滤掉那些置信度低于指定阈值的box。
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX?vd_source=b2eaaddb2c69bf42517a2553af8444ab&spm_id_from=333.788.videopod.episodes&p=16">视频讲解。</a></p>

<a text-decoration="none" href = "https://blog.csdn.net/Keep_Trying_Go/article/details/143696667?sharetype=blogdetail&sharerId=143696667&sharerefer=PC&sharesource=Keep_Trying_Go&spm=1011.2480.3001.8118" >二十七.论文Single-Shot Refinement Neural Network for Object Detection 详解（代码详解）</a><p>
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX?vd_source=b2eaaddb2c69bf42517a2553af8444ab&spm_id_from=333.788.videopod.episodes&p=17">视频讲解。</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143667298" >二十八.论文Scale-Transferrable Object Detection 详解（代码详解）</a><p>提出目的：尺度问题一直是目标检测领域的核心，因为图像或者视频中的物体大小总是不一致的，有大有小，并且有些物体占据的像素特别多，而有的物体占据像素特别少，为了解决这个问题，相关论文也提出了很多算法。
提出方法：提出了尺度变换的目标检测网络，用于多尺度目标的检测。相比于之前的方法，本文提出的方法简化了联合来自不同层多尺度目标的预测。提出的尺度转换模块等同于超分辨率模块，探索了中间尺度一致性，同时也自然的实现跨尺度。最终采用DenseNet作为backbone，尺度转换模块添加到backbone网络最后输出，并且没有带来大的计算量。
<a href = "">视频讲解。</a></p>


<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143508021" >二十九.论文Parallel Feature Pyramid Network for Object Detection 详解（包含代码详解）</a><p>提出目的：最近提出的目标检测方法使用特征金字塔替换了原有的图像金字塔方式，但是目前提出的不同特征层方法限制了其模型的检测表现，为了在应用特征金字塔的同时提升模型性能，本文提出了并行的特征金字塔检测框架。提出方法：本文提出了并行的特征金字塔网络结构，通过拓宽网络模型结构而不是网络模型的深度而构建特征金字塔；首先是应用空间金字塔池化操作和一些额外的特征转换操作生成不同尺度的池化特征图；其次是在并行的特征金字塔模块生成相似语义特征图的特征层；最后是缩放这些池化的特征图到统一大小，并融合统一大小的特征图上下文信息得到最后的特征图。
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX?vd_source=b2eaaddb2c69bf42517a2553af8444ab&p=18&spm_id_from=333.788.videopod.episodes">视频讲解。</a></p>

<a text-decoration="none" href = "https://blog.csdn.net/Keep_Trying_Go/article/details/144109082?sharetype=blogdetail&sharerId=144109082&sharerefer=PC&sharesource=Keep_Trying_Go&spm=1011.2480.3001.8118" ></a><p>三十.关于YOLOv1~YOLOv3源码汇总
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/144010894" >三十一.论文FCOS: Fully Convolutional One-Stage Object Detection详解（代码详解）</a><p>提出目的：当前最新的目标检测算法，比如RetinaNet，SSD，YOLOv3等都是基于anchor box来的，相比于anchor-free的算法，基于anchor box的算法更加复杂，特别是在计算gt box和anchor box之间的最佳匹配关系时。提出方法：本文提出了anchor-free的算法，基于像素级来进行预测（类似语义分割）。不使用anchor box，让整个流程变得更加的简洁，不需要在训练期间计算GT box和anchor box之间的匹配（IOU），并且还省去了设置和anchor box相关的超参数，在anchor-base的算法当中，如果和anchor box相关的超参数设置不好会导致最终的结果差异比较大。本文提出的FCOS（Fully Convolution One-Stage）算法只需要在进行推理的时候会使用到NMS算法。
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/144066826" >三十二.论文Learning Spatial Fusion for Single-Shot Object Detection详解（代码）</a><p>提出目的：特征金字塔被广泛应用于目标检测领域，用于解决多尺度问题，虽然输出不同尺度的特征图用于检测，一定程度提升了检测图像中不同尺度的物体，但是不一致性跨尺度对于特征金字塔是一个主要的限制。提出方法：针对特征金字塔中尺度不一致性问题，本文提出了一种新的，并且以数据驱动的特征金字塔融合策略ASFF（Adaptively Spatial Feature Fusion）。ASFF学习了如何空间过滤冲突信息，以抑制不一致性，从而提高特征的尺度不变性，并几乎没有引入额外的推理开销。通过 ASFF 策略以及稳固的 YOLOv3 为基线实现相关算法。
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143666910" >三十三.论文S3FD: Single Shot Scale-invariant Face Detector 详解（代码详解）</a><p>提出目的：由于已经提出的目标检测算法在小目标的检测方面都普遍比较差，并且本文还是针对人脸进行检测，因此，需要提升当前目标检测算法对于小目标的检测，从而提升在人脸检测方面的性能。。
提出方法：本文提出了一个实时的人脸检测器，基于单张图像尺度不变人脸检测，可以检测不同尺度的人脸，对于小的人脸检测性能也得到很大的提升。
1）提出了一个尺度合理的人脸检测网络，可以很好的检测不同尺度的人脸（基于anchor的人脸检测），并且设计了基于有效感受野的anchor尺度不变和相同间隔比例的原则；
2）通过尺度“补偿”anchor匹配策略，提升了小的人脸检测召回率；
3）通过max-out背景标签的方法，减少了小的人脸中假的正样本。
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/144056101" >三十四.论文Joint Anchor-Feature Refinement for Real-Time Accurate Object Detection in Images and Videos 详解</a><p>提出目的：虽然目标检测算法研究比较多，但是还是存在快速精准定位物体的问题，为了解决这个问题，本文目的是提升目标检测算法在静态或者视频场景中实时准确的检测物体。提出方法：首先，作为一种双重精细化机制，设计了一种新颖的anchor-偏移检测方法，该方法包括锚点精细化、特征位置精细化和可变形检测头。这种新的检测模式能够同时执行两步回归并捕捉准确的物体特征。在 anchor- 偏移检测的基础上，开发了一种双重精细化网络（ DRNet ）用于高性能静态检测，其中进一步设计了一个多可变形头，以利用上下文信息来描述物体（对 RefineDet 的改进）。对于视频中的时间检测，开发了时间精细化网络（ TRNet ）和时间双重精细化网络（ TDRNet ），通过在时间上传播精细化信息。
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/143925085" >三十五.论文An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection详解（代码详解）</a><p>提出目的：DenseNet图像分类模型在分类和目标检测效果上还是不错的，主要是采用了密集连接，但是DenseNet 模型在速度和能量效率上比较低；并且当线性增加通道数的时候，由于密集连接而导致的大量的内存访问，从而导致计算量大和能量消耗变大。提出方法：为了解决DenseNet 密集连接带来的问题，本文提出在每一个block最后才进行连接，而是在最后才进行连接，这样做不仅仅利用了DenseNet 能够在多个感受野情况下表示各种特征的优点，同时也降低了密集连接带来的低效率。实验主要在单阶段的目标检测和两阶段的目标检测算法中进行了验证，实验结果证明本文的思路是有效的。
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://blog.csdn.net/Keep_Trying_Go/article/details/144095444?sharetype=blogdetail&sharerId=144095444&sharerefer=PC&sharesource=Keep_Trying_Go&spm=1011.2480.3001.8118" >三十六.论文FaceBoxes: A CPU Real-time Face Detector with High Accuracy详解（代码）</a><p>提出目的：当前的很多人脸检测算法一般会采用比较大的下采样步长，因为大部分人脸的尺度都比较小，但是目前最大的挑战是实现一个实时并且在CPU上运行的人脸检测算法。提出方法：为了解决上面提到的挑战，本文提出了FaceBoxes人脸检测框架，在速度和准确率上效果都不错。具体方法是：设计一个轻量化的网络模型，包含了RDCL（Rapidly Digested Convolutional Layer）和MSCL（Multiple Scale Convolutional Layers）两个模块。其中RDCL旨在使Face Boxes在CPU上实现实时速度。MSCL旨在丰富不同层次的感受野和离散anchor，以处理不同尺度的面孔。此外，本文提出了一种新的anchor致密化策略，使不同类型的锚点在图像上具有相同的密度，显著提高了小人脸的召回率。
<a href = "https://www.bilibili.com/video/BV1YBeGeNEtX/?vd_source=b2eaaddb2c69bf42517a2553af8444ab">视频讲解。</a></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/144344922" >三十七.论文DenseBox: Unifying Landmark Localization with End to End Object Detection讲解</a>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/144347288" >三十八. 论文SqueezeDet详解</a><p></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/144395120" >三十九.论文ObjectBox: From Centers to Boxes for Anchor-Free Object Detection详解</a><p></p>

<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/144410277" >四十.论文YOLO5Face: Why Reinventing a Face Detector详解</a><p></p>


<a text-decoration="none" href = "https://mydreamambitious.blog.csdn.net/article/details/144428990" >四十一.论文YOLO-FaceV2: A Scale and Occlusion Aware Face Detector详解</a><p></p>


<a text-decoration="none" href = "" >四十二.</a><p></p>

<a text-decoration="none" href = "" >四十三.</a><p></p>

<a text-decoration="none" href = "" >四十四.</a><p></p>

<a text-decoration="none" href = "" >四十五.</a><p></p>

<a text-decoration="none" href = "" >四十六.</a><p></p>

<a text-decoration="none" href = "" >四十七.</a><p></p>

<a text-decoration="none" href = "" >四十八.</a><p></p>

<a text-decoration="none" href = "" >四十九.</a><p></p>




