



# **基于Yolov4训练车辆检测模型**

**模型评估：**模型评估在一张32GTesla V100的GPU上通过'tools/eval.py'测试所有一部分验证集得到，单位是fps(图片数/秒), cuDNN版本是7.5，包括数据加载、网络前向执行和后处理, batch size是1。

**推理时间(fps)**: 推理时间是在一张32GTesla V100的GPU上通过'tools/eval.py'测试所有验证集得到，单位是fps(图片数/秒), cuDNN版本是7.5，包括数据加载、网络前向执行和后处理, batch size是1。

###### Baseline:

模型评估：

| 骨架网络   | 每张GPU图片个数 | fps   | mAP    |
| :--------- | --------------- | ----- | ------ |
| CSPDarkner | 12              | 53.10 | 69.78% |

预测结果：

![下载](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\下载.png)

【评价指标】：

![loss](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\loss.png)

![loss_iou](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\loss_iou.png)



![loss_obj](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\loss_obj.png)



![loss_cls](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\loss_cls.png)



![loss_xy](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\loss_xy.png)



![loss_wh](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\loss_wh.png)



![mAP](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\mAP.png)

1️⃣ 	***改进【一】：*** 将Mish激活函数换为Hardswish激活函数

yolov4的主干网络使用的是Mish激活函数，是一种平滑的非单调的激活函数。

![image-20220720141809626](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220720141809626.png)

Mish激活函数的复杂度较高，计算的同时要引入一个softplus()激活层和另外一个tanh激活函数，从而导致计算速度较慢。

Hardswish激活函数。相较于swish函数，具有数值稳定性好，计算速度快等优点。其数学表达式见公式：

![image-20220720141056224](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220720141056224.png)

hardswish激活函数是对swish激活函数的改进，由公式可见hardswish激活函数可以实现为分段功能，以减少内存访问次数，从而大大降低了等待时间成本。

![image-20220720165925991](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220720165925991.png)

改动前：

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarkner+Mish | PANet+L-relul | 53.10 | 69.78% |

改动后模型评估：

| 骨架网络           | Neck          | fps   | mAP    |
| :----------------- | ------------- | ----- | ------ |
| CSPDarkner+H-swish | PANet+L-relul | 52.06 | 69.76% |

改动前：

![lossa](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\lossa.png)

改动后：

![lossaa](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\lossaa.png)

总结：对比前后两个loss变化图可以看出，改动后训练时间缩短了10分钟左右，loss下降的更快一些；从预测结果来看，预测帧率变化不大，精度变化也不大；综合来看，用Hardswish激活函数替换Mish函数效果并不十分明显。

2️⃣ 	***改进【二】：***  将激活函数L-relul改为H-swish

H-swish下方有界及其无界性，它有助于消除输出神经元的饱和问题，改善网络正则化。它的计算速度比Swish快，并且有利于训练，因为它有助于学习更有表现力的特征，对噪声更鲁棒。

不同的目标检测算法都采用了H-swish激活，大大减少了模型的内存访问次数。这里使用Hard-Swish函数作为Backbone和Neck的主要激活函数，在所考虑的数据集上具有显著的精度增益。此外，提高了检测速度，大大降低了计算成本。

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarkner+Mish | PANet+L-relul | 53.10 | 69.78% |

| 骨架网络           | Neck          | fps   | mAP    |
| :----------------- | ------------- | ----- | ------ |
| CSPDarkner+H-swish | PANet+L-relul | 52.06 | 69.76% |

| 骨架网络           | Neck          | fps   | mAP    |
| :----------------- | ------------- | ----- | ------ |
| CSPDarkner+H-swish | PANet+H-swish | 56.72 | 65.79% |

![lossaaaaa](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\lossaaaaa.png)

![mAPaa](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\mAPaa.png)

总结：可以看到，当我们把Neck的激活函数也换为H-swish后，loss函数的收敛值是比较高的，而且精度也有所下降，可见在yolov4的CSPDarkner、PANet网络中，Mish和L-relul激活函数还是比较好的。

3️⃣	***改进【三】：*** 增加SE注意力机制

SENet 架构论文连接：https://arxiv.org/abs/1709.01507

通过卷积神经网络可以融合通道信息来构建信息特征，通过增加SENet架构可以有效地增加卷积网络的表达能力，“SE注意力机制通过显式建模通道之间的相互依赖关系来自适应地重新校准通道特征响应”，从而能提升网络性能。

![image-20220721221546240](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220721221546240.png)

![image-20220721221604628](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220721221604628.png)

![image-20220721221625993](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220721221625993.png)

在箭头表明的地方增加了SE注意力机制：

![image-20220722110231157](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220722110231157.png)

改动前：

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarkner+Mish | PANet+L-relul | 53.10 | 69.78% |

增加SE后模型评估：

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 33.46 | 70.57% |

loss变化图像：

![lossaaaa](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\lossaaaa.png)

总结：可以看到，增加SE注意力机后模型的训练速度慢了一个小时左右，但是模型评估的精度比Baseline增加了一个点左右（此时我已经把激活函数改回了Mish），但推理的FPS有所下降，可见在卷积网络中增加注意力机制的办法确实可以提升网络性能。（两次修改后预测图片的结果都差不多，所以就没有贴，可能是图片找到不太好，下次换一张图来预测）



4️⃣	***改进【四】：*** 将spp模块换为aspp空洞卷积来增大感受野

在最初的YOLOv4 Backbone中，SPP块与PANet以及CSPDarknet53集成，取代了YOLO其他变体中使用的特征金字塔网络(FPN)。这带来了感受野的显著增加。现在将spp模块换为aspp空洞卷积进一步增增加感受野。

一般认为图片中相邻的像素点存在信息冗余，故而空洞卷积具备以下两个优势：

(1) 扩大感受野：传统的下采样虽可增加感受野，但会降低空间分辨率。而使用空洞卷积能够在扩大感受野的同时，保证分辨率。这十分适用于检测、分割任务中，感受野的增大可检测、分割大的目标，高分辨率则可精确定位目标。
(2) 捕获多尺度上下文信息：空洞卷积中参数 dilation rate 表明在卷积核中填充 (dilation rate-1) 个 0。设置不同 dilation rate 给网络带来不同的感受野，即获取了多尺度信息。

spp是由四个最大池化来提取特征，Sapp通过设置dilation rate和卷积核的大小获得不同的感受野，最后通过拼接融合提取到的特征，获得多尺度信息。

![](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220725154708590.png)

![image-20220725154737551](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220725154737551.png)

将spp换成aspp:

![image-20220725155108500](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220725155108500.png)

• Baseline:

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 53.10 | 69.78% |

• 增加SE后模型评估：

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 33.46 | 70.57% |

• 将spp改为sapp(没加SE):

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 31.59 | 69.86% |

• 将spp改为aspp(加SE):

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 33.08 | 71.97% |

没加SE loss变化图像：

![oss](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\oss.png)

 加SE loss变化图像：

![ss](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\ss.png)

总结：只将spp改为Sapp后对于模型的精度并没有什么提升（与Baseline相比），在loss变化上也想差不大。但是在增加SE注意力机制的基础上再将spp改为sapp后，精度上比Baseline提升了一个点多一点，比只增加SE注意力机制多了不到一个点，可以看到这样改在速度和精度上确实有微小提升。

5️⃣  ***改进【五】：***  修改学习率

学习率可以说是**模型训练最为重要的超参数**。 通常情况下，一个或者一组优秀的学习率既能加速模型的训练，又能得到一个较优甚至最优的精度。 过大或者过小的学习率会直接影响到模型的收敛。因为学习率是官方设置好的，本来没想到要修改学习率，由于改了几个点模型的精度都没有明显的提升，所以我尝试在paddle发布的原版本的基础上修改了'base_lr'=0.001(原来是0.0001)，将'milestones'的范围改为（20000，40000），总iter为60000，将学习率变化的范围分为1:1:1

![image-20220726231557008](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220726231557008.png)

修改后训练完的模型精度竟然达到了85.05%

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 33.91 | 85.05% |

（加SE注意力机制、换成aspp模块）loss变化及精度指标：

![lss](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\lss.png)



<img src="C:\Users\19127\AppData\Roaming\Typora\typora-user-images\mAaP.png" alt="mAaP" style="zoom:67%;" />

总结： 通过适当增加模块，增大感受野，提高特征提取率，修改学习率后，模型精度大幅超过了Baseline，loss的收敛值接近了3，但是最低loss值在大约40000iter左右，之后又有所升高。map达到了85.05%的高精度。可见在修改模型后适当调整学习率会有更好的效果。





6️⃣***改进【六】：*** 精简输出结果，减掉head中的一个输出，在neck中增加yolov5的CSP结构

现有YOLOv4模型的PANet从主干网络分为3层作为输入的。然而，常见对象检测情况与自动驾驶环境不同，有限类别中的物体检测（汽车、行人等，更小的目标也就少了）。基于这个原因，改进PANet可以接收来自**backbone网络**的只有2层的输入。Upsample, Downsample层的位置和数量变少了，计算量相对也就减少，则对应也能减少一个Darknet-53的一个Output Feature,从而减少neck部分的运算，从而使得模型训练和模型预测变得更快。

减少掉一个head输出要从backbone，neck,   head三方面入手，在CSPDaeknet-53主体结构中剪掉DownSample3的输出，也就是剪掉52 * 52的检测图像中小物体的目标框，这样会影响到Neck结构中26 * 26的输出经过卷积和上采样与52 * 52输出的拼接过程，也会影响到head的最后输出结果只有两个，不需要对52 * 52的输出进行3* 3和 1 * 1卷积了，也要调整网络中先验框的数量。

改动大概情况如图标注：

<img src="C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220729160321135.png" alt="image-20220729160321135" style="zoom:200%;" />

通过阅读paddle的CSPdarknet实现代码可知，在Paddle中的backbone与Pytorch版本yolov4-cspbackbone的实现方式不一样，Paddle在backbone中将五次下采样单元的结果全部存放在"blocks[]"列表中，而Pytorch版本yolov4-cspbackbone是分别输出三个所需要的特征图（即52* 52、26* 26、13*13的特征图）。所以要在paddle代码中减少一个输出结构只要在"yolo_head.py"**（**Paddle中将neck部分与head部分的代码全部封装在了"yolo_head.py"中**）**中更改对blocks[]的切片结构以及neck中拼接的循环结构即可；

![image-20220729161249177](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220729161249177.png)

![image-20220729161322682](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220729161322682.png)



| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 49.91 | 79.42% |



![losas](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\losas.png)



![mAasP](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\mAasP.png)

 总结：在head部分剪掉一个输出结构以后精度为79.42%，FPS为  49.91，与上一次训练完85.05%的精度和  33.91的FPS相比，精度有所下降，但是模型推理速度提高了不少，这说明剪掉head的一个输出结果的方法更适合在对精度要求不太高，但是算力不太强的平台上部署，这样更能提高实时监测能力。



 7️⃣***改进【七】：*** 修改CSPResblock网络结构

参考论文：[1]A fast accurate fine-grain object detection model based on YOLOv4 deep neural network

CSPDarknet53中的残差模型帮助网络学习更有表达力的特征，同时减少可训练参数的数量，使其更快地进行实时检测。在原始的YOLOv4模型中，残差单元(Res-unit)进行1×1卷积，然后进行3×3卷积，最后对包含提取的特征信息的两个输出进行权值。在CSPDarknet53网络中，通过卷积操作对输入图像的特征层进行连续降采样，提取细粒度丰富的语义信息。由于最后三层包含相对较高的语义信息，这些信息被传递给SPP和PANet。最后一个特征层包含最好的特征信息，并连接到SPP。其他两层被集成到PANet中，虽然YOLOv4中的残差模块降低了计算成本，但这进一步降低了高分辨率实时检测的计算内存需求。

因此，在CSPDarkNet53网络结构中提出了一个新的残差块，CSP_Block(如图)提高检测速度和性能。

![image-20220727170241003](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220727170241003.png)

#### Part1

残差块的第一部分作为Backbone，在进入主残差单元后进行1×1卷积，再进行3×3卷积，调整通道，为了进一步增强特征提取，然后进行1×1卷积。

#### Part2

而第二部分作为卷积的残差边。在CSP块的末端，这两个部分被连接起来，从而产生额外的特征层信息。

![image-20220809203459455](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220809203459455.png)

![image-20220809203519914](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220809203519914.png)

![image-20220809203538909](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220809203538909.png)

![image-20220809203602891](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220809203602891.png)





| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 52.87 | 84.89% |

（没有加SE注意力机制和aspp模块）loss变化及精度指标：

![laaoss](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\laaoss.png)

![maaAP](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\maaAP.png)



总结：更改后CSP结构相比Baseline的CSP结构更为轻量，在训练结构中可以看出推理速度还是比较高的，而且在没有加SE注意力机制和aspp模块下精度达到了84.89%，可以看出在检测目标较小，密度较大的时候这个CSP结构可以表现得不错。

​	模型推理结果如图：

![下载aa](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\下载aa.png)

Baseline的模型推理结果：

![下载](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\下载.png)

------



8️⃣***改进【八】：*** 用scSE改进yolov4对小目标的检测

在看了相关的博客和论文后，考虑用增加注意力机制增强特征， scSE 注意力本身是将特征图中重要的空间和通道特征增强，使得网络在训练过程中能抓住目标特征的“重点”学习。scSE的网络结构图如下：

![image-20220905164646215](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220905164646215.png)

## 空间注意力sSE

> 1、直接对feature map使用1×1×1卷积, 从[C, H, W]变为[1, H, W]的features
> 2、然后使用sigmoid进行激活得到spatial attention map
> 3、然后直接施加到原始feature map中，完成空间的信息校准

### 通道注意力cSE

> 1、将feature map通过global average pooling方法从[C, H, W]变为[C, 1, 1]
> 2、然后使用两个1×1×1卷积进行信息的处理，最终得到C维的向量
> 3、然后使用sigmoid函数进行归一化，得到对应的mask
> 4、最后通过channel-wise相乘，得到经过信息校准过的feature map

scSE就是将sSE和cSE相加起来。

### 添加scSE注意力：

![image-20220905164908224](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220905164908224.png)

将scSE加在：

![image-20220905165209141](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220905165209141.png)



加scSE后：

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 45.39 | 84.39% |



![loss (1)](C:\Users\19127\Downloads\loss (1).png)

![mAaPddd](C:\Users\19127\Desktop\mAaPddd.png)



预测结果：（基本和前一个改进点一样）

![下载aa](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\下载aa.png)

另一张图片：

![image-20220905172307105](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220905172307105.png)

总结：在yolov4模型上，yolov4本身可能并不适合小目标的目标检测，小目标由于分辨率低、体积小，很难被检测到。而且小目标检测性能主要是由于网络模型的局限性和训练数据集的不平衡所造成的。对于图一中后面的模糊车辆可能由于训练时数据集的选取没有很多类似的数据导致训练出来的模型在加了一些改动后并不能很好的识别，但是在比较清晰一些的检测图片上，小的车辆目标还是可以检测得到目标框的。

------

------



实验记录总结表：

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarkner+Mish | PANet+L-relul | 53.10 | 69.78% |

| 骨架网络           | Neck          | fps   | mAP    |
| :----------------- | ------------- | ----- | ------ |
| CSPDarkner+H-swish | PANet+L-relul | 52.06 | 69.76% |

| 骨架网络           | Neck          | fps   | mAP    |
| :----------------- | ------------- | ----- | ------ |
| CSPDarkner+H-swish | PANet+H-swish | 56.72 | 65.79% |

• 增加SE后模型评估：

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 33.46 | 70.57% |

• 将spp改为sapp(没加SE):

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 31.59 | 69.86% |

• 将spp改为sapp(加SE):

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 33.08 | 71.97% |

修改后训练完的模型精度竟然达到了85.05%

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 33.91 | 85.05% |

剪掉一个head

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 49.91 | 79.42% |

修改CSP网络结构后

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 52.87 | 84.89% |

加scSE后：

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 45.39 | 84.39% |

总结：综合以上改动，最高的精度达到了85.05%，最高的FPS达到了  56.72，综合这两个指标来看，最好的模型在修改CSP网络结构后，兼顾了精度与速度：

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 52.87 | 84.89% |

