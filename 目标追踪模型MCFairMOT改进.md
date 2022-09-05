

# **基于PaddleDetection的中的MCFairMOT算法实现的目标追踪**

## FairMOT：论多目标跟踪中检测和再识别的公平性

[张毅夫](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y)， [王春玉](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C)， [王兴刚](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X)， [曾文军](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+W)， [刘文宇](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+W)

> 多目标跟踪（MOT）是计算机视觉中的一个重要问题，具有广泛的应用。将MOT表述为在单个网络中进行对象检测和重新ID的多任务学习是很有吸引力的，因为它允许两个任务的联合优化，并且具有很高的计算效率。然而，我们发现这两项任务往往相互竞争，需要谨慎处理。特别是，以前的作品通常将re-ID视为次要任务，其准确性受到主要检测任务的严重影响。结果，网络偏向于主要检测任务，这对重新识别任务不公平。为了解决这个问题，我们提出了一种简单而有效的方法，称为FairMOT，它基于无锚点对象检测架构CenterNet。请注意，它不是 CenterNet 和 re-ID 的天真组合。相反，我们提出了一堆详细的设计，这些设计对于通过彻底的实证研究获得良好的跟踪结果至关重要。由此产生的方法实现了检测和跟踪的高精度。该方法在多个公共数据集上的表现远远超过最先进的方法。

### **算法介绍：**

FairMOT以Anchor Free的CenterNet检测器为基础，克服了Anchor-Based的检测框架中anchor和特征不对齐问题，深浅层特征融合使得检测和ReID任务各自获得所需要的特征，并且使用低维度ReID特征，提出了一种由两个同质分支组成的简单baseline来预测像素级目标得分和ReID特征，实现了两个任务之间的公平性，并获得了更高水平的实时多目标跟踪精度。

MCFairMOT是[FairMOT](https://arxiv.org/abs/2004.01888)的多类别扩展版本。

•***采用 MCFairMOT 模型作为基线模型***

[MOTChallenge](https://motchallenge.net/)是多目标跟踪领域最为常用的benchmark，以MOTChallenge中的评价标准：

MOTA的描述：MOTA给出了一个非常直观的衡量跟踪器在检测物体和保持轨迹时的性能，与物体位置的估计精度无关。

IDF1的描述:  引入track ID的F1

###### Baseline:

【训练评价指标】：

![loss3](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\loss3.png)



![offset_loss3](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\offset_loss3.png)

![reid_loss3](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\reid_loss3.png)

![size_loss3](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\size_loss3.png)

![det_loss3](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\det_loss3.png)

![heatmap_loss3](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\heatmap_loss3.png)



模型评估：

| 骨架网络 | 训练策略 | 推理时间(s) | IDF1  | FPS   | mOTa  | 召回率 |
| :------- | -------- | ----------- | ----- | ----- | ----- | ------ |
| DLA-34   | 30e      | 27.57       | 61.2% | 15.12 | 51.3% | 60.5%  |

模型预测1：

第一次预测：

![捕获](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\捕获.PNG)

1️⃣ 	***改进【一】：***

baseline使用的是Momentum优化器,学习率是神经网络训练中最重要的超参数之一,使用Warmup预热学习率的方式,即先用最初的小学习率训练，然后每个step增大一点点，直到达到最初设置的比较大的学习率时（注：此时预热学习率完成），采用最初设置的学习率进行训练（注：预热学习率完成后的训练过程，学习率是衰减的），有助于使模型收敛速度变快，效果更佳。

需要修改配置文件`mcfairmot_dla34_30e_1088x608_visdrone_vehicle.yml`中的`LearningRate`和`OptimizerBuilder`字段，修改后的内容如下：

```yaml
LearningRate:
  base_lr: 0.01 #学习率决定了权值更新的速度，学习率大，更新的就快，但太快容易越过最优值，而学习率太小又更新的慢，效率低，一般学习率随着训练的进行不断更改，先高一点，然后慢慢降低
  schedulers:
  - !PiecewiseDecay
    gamma: 0.1 #(用来设置衰减比率)
    milestones: [15, 22] #(学习率变化界限，即学习率变动因子：如迭代到15次时，学习率衰减十倍，22次迭代时，学习率又会在前一个学习率的基础上衰减十倍)
    use_warmup: True #(使用预热学习率)
  - !BurninWarmup
    steps: 1000 #(学习率变动步长)

OptimizerBuilder:  #（设置优化器）
  optimizer:
    type: Momentum
    #正则化
  regularizer:
    factor: 0.0001 #（重衰减正则系数，防止过拟合）
    type: L2
```

loss下降过程：

![loss2](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\loss2.png)



 ![heatmap_loss1](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\heatmap_loss1.png)



<img src="C:\Users\19127\AppData\Roaming\Typora\typora-user-images\det_loss1.png" alt="det_loss1" style="zoom:200%;" />



![size_loss1](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\size_loss1.png)



![reid_loss1](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\reid_loss1.png)



![offset_loss1](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\offset_loss1.png)

模型评估：

|          | 骨架网络 | 训练策略 | 推理时间(s) | IDF1  | FPS   | mOTa  | 召回率 |
| -------- | :------- | -------- | ----------- | ----- | ----- | ----- | ------ |
| Baseline | DLA-34   | 30e      | 27.57       | 61.2% | 15.12 | 51.3% | 60.5%  |

|              | 骨架网络 | 训练策略 | 推理时间(s) | IDF1  | FPS   | mOTa  | 召回率 |
| ------------ | :------- | -------- | ----------- | ----- | ----- | ----- | ------ |
| 优化学习率后 | DLA-34   | 30e      | 25.32       | 63.3% | 20.67 | 52.1% | 62.6%  |

模型预测：![Inked下载 (1)](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\Inked下载 (1).jpg)

![Inked下载1](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\Inked下载1.jpg)

总结：

增加啊学习率设置和增加优化器后从loss变化值中可以看出，loss收敛后趋近的数值远小于修改之前的趋近值，在视频检测的结果中可以看到“白色货车”也可以被准确的识别出来了，检测和追踪的实际效果也有明显的提升。

针对现在的实验效果来看，还有一下问题：

•  模型的速度和精度还需要进一步优化；

•  识别效果不够准确（例如图中标记的红三角部分）；

•  存在一定的漏检车辆，导致召回率较低；

2️⃣***改进【二】：***  更换骨架网络--HRNet-W18

HRNetV2-W18网络的论文地址：https://arxiv.org/pdf/1904.04514.pdf

#####  网络结构:

1.先是两个stride=2的3x3卷积将输入图像分辨率降采样至1/4。

2.stage 1有4个residual单元，每个单元由width为64的bottleneck和紧随其后的一个3x3卷积将feature map的width减少为C组成，这里的width其实就是通道数。

3.stage 2，stage 3和stage 4分别包含1，4和3个多分辨块。对于每个多分辨块的四个不同分辨率的卷积，其通道数分别为C，2C，4C和8C。

4.多分辨群卷积的每一个分支包含4个residual单元，每个单元在每个分辨率中包含两个3x3卷积。

轻量模型的骨架网络可以加快网络运行速度，这里将dla34网络换为更加轻量的HRNetV2-W18。HRNetV2-W18在HRNetV1的基础上增加了对低分辨率的卷积并表示，在原论文做的消融实验中并没有与dla-34网络进行比较，查看paddlepaddle的代码后，paddle已经将HRNetV2-W18网络封装好了，所以只需更改配置文件，新增fairmot_hardnet18_30e_1088x608.yml文件：

```yml
_BASE_: [
  '../../datasets/mot.yml',
  '../../runtime.yml',
  '_base_/optimizer_30e.yml',
  '_base_/fairmot_hardnet18.yml',
  '_base_/fairmot_reader_1088x608.yml',
]

weights: output/fairmot_hardnet18_30e_1088x608/model_final
```

模型评估：

|          | 骨架网络 | 训练策略 | 推理时间(s) | IDF1  | FPS   | mOTa(准确率) | 召回率 |
| -------- | :------- | -------- | ----------- | ----- | ----- | ------------ | ------ |
| Baseline | DLA-34   | 30e      | 27.57       | 41.4% | 15.12 | 26.3%        | 45.6%  |

|              | 骨架网络 | 训练策略 | 推理时间(s) | IDF1  | FPS   | mOTa(准确率) | 召回率 |
| ------------ | :------- | -------- | ----------- | ----- | ----- | ------------ | ------ |
| 优化学习率后 | DLA-34   | 30e      | 20.32       | 63.3% | 20.67 | 52.1%        | 62.6%  |

|          | 骨架网络    | 训练策略 | 推理时间(s) | IDF1  | FPS   | mOTa(准确率) | 召回率 |
| -------- | :---------- | -------- | ----------- | ----- | ----- | ------------ | ------ |
| 更改网络 | HRNetV2-W18 | 30e      | 25.32       | 63.3% | 14.44 | 75.1%        | 59.6%  |

![image-20220723093755744](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\image-20220723093755744.png)

loss变化：

![losa](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\losa.png)



![reid_lossa](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\reid_lossa.png)

总结：

从loss变化图中可以看到，更改网络后的loss趋近于1.5的下方，reid_loss大概趋近于4，而修改之前loss趋近于1.8的下方，reid_loss大概趋近于10.5。可以看出相关loss的值越来越小，误差越来越小，印证了更换网络后识别和跟踪准确率的提高；更换backbone为HRNetV2-W18在速度和精度上都有了显著提升；在实际检测图上可以看到右上角刚刚出现的车辆也能及时检测到，检测效果有明显的提升。



3️⃣***改进【三】：***   使用ByteTracker 

BYTE是一种非常简单且有效的数据关联方法，其主要思想是利用检测框和跟踪轨迹之间的相关性，在保留高置信度检测框的基础上，挖掘低置信度检测框中真实的目标，从而实现降低漏检并提高跟踪轨迹连贯性的作用。具体做法为：

对于高置信度的检测框，跟之前的跟踪轨迹进行匹配；对于低置信度的检测框，跟1中剩余的轨迹进行二次关联，将所有不匹配的低分数检测框视为背景并删除；对于没有关联上的轨迹，保留30帧，如果一直没有检测框与其进行关联，则删除该轨迹。

使用 ByteTracker 可以通过更改配置文件中的`JDETracker`字段来实现，修改后的内容如下：

```
JDETracker:
  use_byte: True
  match_thres: 0.8
  conf_thres: 0.4
  low_conf_thres: 0.1
  min_box_area: 0
  vertical_ratio: 0
```

|                 | 骨架网络 | 训练策略 | 推理时间(s) | IDF1  | FPS   | mOTa(准确率) | 召回率 |
| --------------- | :------- | -------- | ----------- | ----- | ----- | ------------ | ------ |
| 使用ByteTracker | DLA-34   | 30e      | 23.32       | 57.3% | 18.44 | 39.1%        | 62.6%  |

|                 | 骨架网络    | 训练策略 | 推理时间(s) | IDF1  | FPS   | mOTa(准确率) | 召回率 |
| --------------- | :---------- | -------- | ----------- | ----- | ----- | ------------ | ------ |
| 使用ByteTracker | HRNetV2-W18 | 30e      | 16.94       | 71.8% | 24.76 | 56.0%        | 68.6%  |

![s](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\s.png)

![eid_loss](C:\Users\19127\AppData\Roaming\Typora\typora-user-images\eid_loss.png)

总结：可以看到，对于不同的backbone，使用ByteTracker进行优化，能够明显提升模型的召回率，改善漏检现象，从而取得IDF1的提升，同时几乎不会带来多少速度损失。





