# Day 1

## 学习常用的网络结构：

### 1.残差网络：

在深层卷积神经网络中，梯度优化更困难，残差网络将输入值x经过权重层和激活函数得到的结果与原来的x的值相加，在经过一个激活函数得到输出结果，解决了难以训练的问题，提高训练速度。具体有两种方式：

![image-20220629122704349](https://github.com/nongfulv2/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220629122704349.png)
左图对应输入通道为64时，用两个(3,3)的卷积，中间不改变通道数，最后**相加激活**得到输出。右图对应于输入通道为256时，先用一个(1,1)的卷积把**通道数缩小为1/4**，然后在这个通道数上进行(3,3)的卷积，最后再经过一个(1,1)的卷积改为原通道，然后**相加激活**的到输出。

**ResNet34中使用的是左图的残差结构，ResNet50/101/152使用的是右图的残差结构。**

### 2.深度可分离卷积：

在**轻量化的卷积网络**中应用广泛。它的结构如下图：

![image-20220629123205219](https://github.com/nongfulv2/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220629123205219.png)

对于一个输入矩阵，我先用一个(3,3)的卷积，各个通道**不相加**，这个过程其实就是**分组卷积**，**分组的个数等于输入通道数**，之后再用(1,1)的卷积改变**输出通道数**。

深度可分离卷积**极大地降低了卷积的参数量**。

### 3.SE注意力机制：

**SE注意力机制**希望模型可以**自动学习到不同channel特征的重要程度**，因为很显然各个通道间**所包含特征的重要程度是不一样的**。它的结构如下：

![image-20220629123820679](https://github.com/nongfulv2/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220629123820679.png)

通过一个**全局平均池化**，将输入的大小压缩成1*1，通道数不变，对这个一维矩阵做激励，得到的这样的一个一维矩阵相当于是各个通道的权重，将这个一维矩阵与原输入相乘，得到一个重要成不同的输出，这样就区分出了各个通道的重要性。

原始SE注意力机制的**压缩**和**激励**如下：

![image-20220629124755627](https://github.com/nongfulv2/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220629124755627.png)

对**ResNet**和**MobileNet**使用**注意力机制**：

![image-20220629130544659](https://github.com/nongfulv2/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220629130544659.png)

**ResNet**中的**SE注意力机制**放在了(1,1)卷积之后，激励部分使用的是**全连接层**。MobileNetV3中**SE注意力机制**放在了(1,1)卷积之前，并且激励部分使用的是**卷积层**。

------



## 论文泛读——方面级情感三元组的分析：

抽取方面词、观点词、关联情感--->跨域共享联合抽取(从句子中端到端提取aspect情感三元组)

### 1.跨域共享联合抽取：

首先利用基编码器学习单词级表示，并利用图卷积网络在句子的句法依存树上获取单词的依存信息，融合到跨度表示中；然后，列举并过滤句子中所有可能的跨度，生成候选跨度，这些候选跨度可以作为方面词和观点词共享；最后，将两个分别作为方面词和观点词的候选词及其对应的局部上下文输入分类器，一次性生成情感三元组。

### 2.GCN（图卷积神经网络）

解决图结构数据，不同于CNN、RNN模型，但是和CNN的本质一样，都是特征提取器，只不过他的对象是图数据。GCN用这些特征进行节点分类，图分类，边预测。

什么结构：有N个节点的N*D维节点特征组成的矩阵X和   各个节点之间的关系N * N维的矩阵A加上单位矩阵I在进行归一化    相乘，加上一个权重，再激活一下。

**即使不训练，完全使用随机初始化的参数W，GCN提取出来的特征就以及十分优秀了！**这跟CNN不训练是完全不一样的，后者不训练是根本得不到什么有效特征的。


# Day 2




MobilenetV3轻量级网络。

主要结构：

1.深度可分离卷积

2.SE注意力机制、

3.新型的激活函数：Hardswish激活函数
![image-20220630183931997](https://github.com/nongfulv2/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220630183931997.png)

Hardswish函数在梯度上存在突变，对于训练的精度不利，但是在深层网络中影响较小，所以可以在浅层网络中可以用Relu函数代替。

4.修改网络结构：缩减卷积层并没有减少太多精度
从MobilenetV2到MobilenetV3的结构改动：V2使用了四层卷积再接一个平均池化；V3只用一个卷积层修改通道数直接接了平均池化层，大大减少网络参数量。
整个的V3网络是一个残差网络结构，内嵌SE注意力机制，通过1* 1卷积改变通道数.
![image-20220702123331510](https://github.com/nongfulv2/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220702123331510.png)


# Day 3


系统复习YOLO v3网络结构并总结，为便于复习查阅，见单独文件



# Day 4



使用Ai studio 增加算力，创建新项目并总结笔记，为便于复习查阅，见单独文件



# Day 5



## 评价指标

### 1.精确率（Precision）与召回率（Recall）

![img](https://img-blog.csdnimg.cn/20191109225656130.png)

（ 准确率一般用来评估模型的全局准确程度，无法全面评价一个模型性能。）

TP:正样本被预测成个样本

TN：负样本被预测成负样本

FP：负样本被预测成正样本

FN：正样本被预测成负样本

P（精确率、查准率）：预测为正样本中实际为正样本的占比。

R（召回率、查全率）：所有正样本中实际被预测为正样本的占比。

### 2.P-R曲线（查准率-查全率曲线）：

![img](https://img-blog.csdnimg.cn/img_convert/3b5947b4441f3bd51b48a67208bdf0fa.png)

在进行比较时，若一个学习器的 P-R曲线被另外一个学习器的曲线完全“包住”，则可断言后者的性能优于前者。P-R曲线下面积的大小，在一定程度上表征了学习器在查准率和查全率上取得相对“双高”的比例。

AP是PR曲线下面的面积，一个越好的分类器，AP值越高。

如果是多类别目标检测任务，就要使用mAP，mAP是多个类别AP的平均值。这个mean的意思是对每个类的AP再求平均，得到的就是mAP的值，mAP的大小一定在[0,1]区间，越大越好。该指标是目标检测算法中最重要的一个。

改进方式：
- H-Swish与大卷积核可以提升模型性能且不会造成较大的推理损耗；
- 在网络的下层添加少量的SE模块可以更进一步提升模型性能且不会产生过多的损耗（实际上Lcnet仅仅这是在最后两层添加注意力，但是提升效果明显）；
- AP后采用更大FC层可以极大提升模型性能（但也会让模型参数和计算量暴涨）；
- dropout技术可以进一步提升了模型的精度


目前打算的改进方法：
**对于YOLOv3的改进：通过引入可变形卷积，dropblock，IoU loss和Iou aware，将精度进一步提升；使用label smooth（标签平滑）提高模型的泛化性能和准确率；改变可以增加注意力机制；使用数据增强；将Darknet-53的比较深的网络（参数量非常庞大）改为轻量化网络；精简输出结果，减掉head中的一个输出，则对应也能减少一个Darknet-53的一个Output Feature,从而减少neck部分的运算；对先验框调整，原始是3个，可以改为2个或4个。**



# Day 6


YOLOv3的损失函数理解:

def obj_loss(self, pbox, gbox, pobj, tobj, anchor, downsample):
        
        pbox = decode_yolo(pbox, anchor, downsample)
        pbox = xywh2xyxy(pbox)
        pbox = paddle.concat(pbox, axis=-1)
        b = pbox.shape[0]
        pbox = pbox.reshape((b, -1, 4))
   
        gxy = gbox[:,_, 0:2] - gbox[:, :, 2:4] * 0.5
        gwh = gbox[:, :, 0:2] + gbox[:, :, 2:4] * 0.5
        gbox = paddle.concat([gxy, gwh], axis=-1)


        iou = iou_similarity(pbox, gbox)
        iou.stop_gradient = True
        iou_max = iou.max(2)  # [N, M1]
        iou_mask = paddle.cast(iou_max <= self.ignore_thresh, dtype=pbox.dtype)
        iou_mask.stop_gradient = True

        pobj = pobj.reshape((b, -1))
        tobj = tobj.reshape((b, -1))
        obj_mask = paddle.cast(tobj > 0, dtype=pbox.dtype)
        obj_mask.stop_gradient = True

        loss_obj = F.binary_cross_entropy_with_logits(
            pobj, obj_mask, reduction='none')
        loss_obj_pos = (loss_obj * tobj)
        loss_obj_neg = (loss_obj * (1 - obj_mask) * iou_mask)
        return loss_obj_pos + loss_obj_neg

    def cls_loss(self, pcls, tcls):
        if self.label_smooth:
            delta = min(1. / self.num_classes, 1. / 40)
            pos, neg = 1 - delta, delta
            # 1 for positive, 0 for negative
            tcls = pos * paddle.cast(
                tcls > 0., dtype=tcls.dtype) + neg * paddle.cast(
                    tcls <= 0., dtype=tcls.dtype)

        loss_cls = F.binary_cross_entropy_with_logits(
            pcls, tcls, reduction='none')
        return loss_cls

    def yolov3_loss(self, p, t, gt_box, anchor, downsample, scale=1.,
                    eps=1e-10):
        na = len(anchor)
        b, c, h, w = p.shape
        if self.iou_aware_loss:
            ioup, p = p[:, 0:na, :, :], p[:, na:, :, :]
            ioup = ioup.unsqueeze(-1)
        p = p.reshape((b, na, -1, h, w)).transpose((0, 1, 3, 4, 2))
        x, y = p[:, :, :, :, 0:1], p[:, :, :, :, 1:2]
        w, h = p[:, :, :, :, 2:3], p[:, :, :, :, 3:4]
        obj, pcls = p[:, :, :, :, 4:5], p[:, :, :, :, 5:]
        self.distill_pairs.append([x, y, w, h, obj, pcls])

        t = t.transpose((0, 1, 3, 4, 2))
        tx, ty = t[:, :, :, :, 0:1], t[:, :, :, :, 1:2]
        tw, th = t[:, :, :, :, 2:3], t[:, :, :, :, 3:4]
        tscale = t[:, :, :, :, 4:5]
        tobj, tcls = t[:, :, :, :, 5:6], t[:, :, :, :, 6:]

        tscale_obj = tscale * tobj
        loss = dict()

        x = scale * F.sigmoid(x) - 0.5 * (scale - 1.)
        y = scale * F.sigmoid(y) - 0.5 * (scale - 1.)

        if abs(scale - 1.) < eps:
            loss_x = F.binary_cross_entropy(x, tx, reduction='none')
            loss_y = F.binary_cross_entropy(y, ty, reduction='none')
            loss_xy = tscale_obj * (loss_x + loss_y)
        else:
            loss_x = paddle.abs(x - tx)
            loss_y = paddle.abs(y - ty)
            loss_xy = tscale_obj * (loss_x + loss_y)

        loss_xy = loss_xy.sum([1, 2, 3, 4]).mean()

        loss_w = paddle.abs(w - tw)
        loss_h = paddle.abs(h - th)
        loss_wh = tscale_obj * (loss_w + loss_h)
        loss_wh = loss_wh.sum([1, 2, 3, 4]).mean()

        loss['loss_xy'] = loss_xy
        loss['loss_wh'] = loss_wh

        if self.iou_loss is not None:
            # warn: do not modify x, y, w, h in place
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou = self.iou_loss(pbox, gbox)
            loss_iou = loss_iou * tscale_obj
            loss_iou = loss_iou.sum([1, 2, 3, 4]).mean()
            loss['loss_iou'] = loss_iou

        if self.iou_aware_loss is not None:
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou_aware = self.iou_aware_loss(ioup, pbox, gbox)
            loss_iou_aware = loss_iou_aware * tobj
            loss_iou_aware = loss_iou_aware.sum([1, 2, 3, 4]).mean()
            loss['loss_iou_aware'] = loss_iou_aware

        box = [x, y, w, h]
        loss_obj = self.obj_loss(box, gt_box, obj, tobj, anchor, downsample)
        loss_obj = loss_obj.sum(-1).mean()
        loss['loss_obj'] = loss_obj
        loss_cls = self.cls_loss(pcls, tcls) * tobj
        loss_cls = loss_cls.sum([1, 2, 3, 4]).mean()
        loss['loss_cls'] = loss_cls
        return loss



（标签平滑）label smooth regularization作为一种简单的正则化技巧，它能提高分类任务中模型的泛化性能和准确率，缓解数据分布不平衡的问题。简单来说就是神经网络在交叉熵损失函数的时候，随着loss逐渐降低，使真样本为1，负样本为0，这样过于绝对，有了label smooth就会使前者趋于1，后者趋于0，能够提升模型效果。


loss_obj(置信度损失):（没看明白......）过两天继续理解
loss_cls(分类损失)：也用的是二值交熵，但是 “paddle.nn.functional.binary_cross_entropy_with_logits”结合了 sigmoid 操作和 api_nn_loss_BCELoss 操作。同时，我们也可以认为这个函数是 sigmoid_cross_entrop_with_logits和一些 reduce 操作的组合，所以这里不用对cls做一次sigmoid()操作和reduce操作。

loss_x,loss_y（中心坐标）:用的是二值交叉熵损失， "paddle.nn.functional.binary_cross_entropy"函数用于计算输入 input和标签 label 之间的二值交叉熵损失值，输入通常为sigmoid的输出，所以loss_x，loss_y的输入x,y均经过了sigmoid激活函数；tscal是一个面积权重，真样本时tobj=1，负样本使为0，tscale_obj = tscale *tobj，这样就只对正样本计算了损失，负样本不会计算x损失，因为负样本的tobj处是0，0乘任何数都得0，也就是说负样本的x损失是0，而常数0的导数是0，也就是说负样本不会得到梯度。loss_xy就得到了预测的中心点坐标x,y求平均后的损失值；

loss_w,loss_h（预测框的长宽）:用的是绝对损失，也就是L1损失，loss_wh同样也乘了tscale_obj,只计算真样本的损失，再求平均；

iou损失，这一项是为了辅助监督预测框的坐标和大小，作为xywh损失的补充。它同样乘上了tscale_tobj，即只计算正样本的损失。iou损失，即我们希望预测框和gt之间的iou尽可能大，iou即交并比。计算iou损失时，就真的需要把上述的xywh解码成bx by bw bh再和gt框计算iou损失，求平均。iou_aware_loss更能提升精度；



# Day 7



YOLO v3所有模型均在VOC2007数据集中训练和测试。

ImageNet预训练模型：Paddle提供的基于ImageNet的骨架网络预训练模型。

**模型评估：**模型评估在一张32GTesla V100的GPU上通过'tools/eval.py'测试所有一部分验证集得到，单位是fps(图片数/秒), cuDNN版本是7.5，包括数据加载、网络前向执行和后处理, batch size是1。

**推理时间(fps)**: 推理时间是在一张32GTesla V100的GPU上通过'tools/eval.py'测试所有验证集得到，单位是fps(图片数/秒), cuDNN版本是7.5，包括数据加载、网络前向执行和后处理, batch size是1。

###### Baseline:

模型评估：

| 骨架网络   | 每张GPU图片个数 | 学习率策略 | 推理时间(fps) | mAP    |
| :--------- | --------------- | ---------- | ------------- | ------ |
| Darkner_53 | 12              | 270e       | 38.39         | 68.41% |

![loss](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/loss%20(1).png)

![loss_obj (1)](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/loss_cls.png)

![loss_wh](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/loss_obj%20(1).png)

![loss_xy](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/loss_wh%20(1).png)

![loss](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/loss_xy%20(1).png)

模型预测1：

|         | 预测时间（ms） | 图片数量 |      |      |      |
| ------- | -------------- | -------- | ---- | ---- | ---- |
| street  | 1818.2         | 1        |      |      |      |
| street1 | 2072.8         | 1        |      |      |      |

第一次预测：

street:

![image-20220708203852435](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220705165333625.png)
street1:
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220705165411068.png)



# Day 8


增加IOU-aware（论文阅读总结）：

论文地址：[1912.05992.pdf (arxiv.org)](https://arxiv.org/pdf/1912.05992.pdf)

论文名称：IoU-aware Single-stage Object Detector for Accurate Localization

这篇文章提出了classification score和IoU之间不相关的问题，通过预测detected boxes和Ground Truth之间IoU，并在Inference阶段同时使用预测的IoU和classification score得到一个新的检测置信度，在新的检测置信度基础上进行NMS和AP的计算。

作者指出目前single-stage检测器存在的问题：**classification score和localization accuracy之间的低相关性使得模型性能无法得到提升**，为了解决这个问题，提出了IoU-aware single-stage object detector。

在这项工作中，**作者说明了single-stage object detection在Classification score和Localization Quality之间的低相关性会严重损害模型的性能(mAP)**。因此，作者提出的**IoU-aware single-stage object detector**是通过在regression分支的最后一层添加IoU Prediction head来预测每个detected box 的IoU。使得模型知道每个detected box的localization Quality。在推断过程中，通过**将分类分数和预测IoU相乘最为最终的检测置信度**，然后在随后的NMS和AP计算中使用该置信度对所有检测进行排序。

所以，在源码Paddledetection_tutorial/ppdet/modeling/heads/yolo_head.py文件下修改iou_aware为True，在最后的输出上，增加了一个IoU Prediction Head，用于预测IoU。并在inference阶段使用Classification score乘上预测IoU作为最终置信度分数，更好地提升了精度。

mAP:71.13%		FPS:39.1

street:1810.8
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/tempsnip.png)
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/tempsnip1.png)


iou loss改为Ciou loss：

论文地址：[Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression (arxiv.org)](https://arxiv.org/pdf/1911.08287.pdf)

论文名称：Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression

论文阅读总结：DIoU Loss将两个框之间的距离作为损失函数的惩罚项，比GIoU Loss先增加预测框的大小使其与目标重叠收敛快得多，所以拥有比GIoU Loss更好的表现。CIOU在DIOU的基础上将Bounding box的纵横比考虑进损失函数中，进一步提升了回归精度。


# Day 9

iou loss改为Ciou loss：

论文地址：[Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression (arxiv.org)](https://arxiv.org/pdf/1911.08287.pdf)

论文名称：Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression

论文阅读总结：DIoU Loss将两个框之间的距离作为损失函数的惩罚项，比GIoU Loss先增加预测框的大小使其与目标重叠收敛快得多，所以拥有比GIoU Loss更好的表现。CIOU在DIOU的基础上将Bounding box的纵横比考虑进损失函数中，进一步提升了回归精度。


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 听了部长的一番淳淳教导改做YOLOv5 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`


# Day 11
# YOLOV5  学习笔记记录

## 1 常用归一化手段

### 1.1 BN、GN、IN和LN

这4种归一化手段网上分析文章非常多，学习了之后简单记下了相关概要及心得


假设输入维度是(N,C,H,W),不管哪一层归一化手段，都不会改变输出大小，即输出维度也是(N,C,H,W)。

**(1) BN**

对于BN，其归一化维度是N、HxW维度，故其可学习权重维度是(C,)，**其实就是BN的weight和bias维度**。BN本质意思就是在Batch和HxW维度进行归一化，可以看出和batch相关，如果batch比较小，那么可能统计就不准确。并且**由于测试时候batch可能和训练不同，导致分布不一致，故还多了两个参数：全局统计的均值和方差值**



**(2) LN**

对于LN，**其归一化维度是C、HxW维度或者HxW维度或者W维度，但是不可以归一化维度为H**，可以设置，比较灵活，其对每个batch单独进行各自的归一化操作，归一化操作时候不考虑batch，所以可以保证训练和测试一样。 例如：

```python
m = nn.LayerNorm(normalized_shape=[100 ,35 ,45])
input = torch.randn(20, 100, 35, 45)
```

其可学习权重维度是(100,35,45)：**对batch输入计算均值和方差(C、H和W维度求均值)，输出维度为(N,)，然后对输入(N,C,H,W)采用计算出来的(N,)个值进行广播归一化操作，最后再乘上可学习的(C,H,W)个权重参数即可**。

**(3) IN**

对于IN，其归一化维度最简单，就是HxW，如下所示：


输入参数必须且只能是C，其内部计算是：**对batch输入计算均值和方差(H,W维度求均值方差)，输出维度为(N,C),然后对输入(N,C,H,W)采用计算出来的(N,C)个值进行广播归一化操作，最后再乘上可学习的(C,)个权重参数即可**。


**(4) GN**

GN是介于LN和IN之间的操作，多了一个group操作，

其内部计算是：**对batch输入计算均值和方差(C/组数、H,W维度求均值方差)，输出维度为(N,组数),然后对输入(N,C,H,W)采用计算出来的(N,组数)个值进行广播归一化操作，最后再乘上可学习的(C,)个权重参数即可**。不需要强制开启eval模式。

# Day 13
### （对YOLOv5中FRN的理解）
## 1.2 FRN

虽然GN解决了小batch size时如果batch比较小，那么可能统计就不准确的问题，但在正常的batch size时，其精度依然比不上BN层。FRN解决了归一化既不依赖于batch，又能使精度高于BN。FRN层由两部分组成，Filtere Response Normal-ization (FRN)（过滤响应归一化）和Thresholded Linear Unit (TLU)（阈值线性单元）。

**(1) FRN**

N是HxW,表面看起来计算方式非常类似IN，计算过程是**：对输入的batch个样本在HxW维度上计算方差，不计算均值，得到输出维度(batch,c)，然后对(batch,c,h,w)进行除方差操作，并且引入可学习参数，权重维度是(C,),最后对上述输出乘以可学习参数即可输出**。

**(2) TLU**

由于在FRN操作中**没有减去均值**，会导致“归一化”后的特征值不是关于零对称，可能会以任意的方式偏移零值。如果使用ReLU作为激活函数的话，会引起误差，产生很多零值，性能下降。所以需要对ReLU进行增强，即TLU，引入一个可学习的阈值τ。

# Day14
yolov5相比于yolov4，在模型方面最大特点是灵活，其引入了depth_multiple和width_multiple系数来得到不同大小模型：

```text
yolov5s: 
    depth_multiple: 0.33
    width_multiple: 0.50
yolov5m: 
    depth_multiple: 0.67
    width_multiple: 0.75
yolov5l: 
    depth_multiple: 1.0
    width_multiple: 1.0
```

depth_multiple表示channel的缩放系数，就是将配置里面的backbone和head部分有关通道的设置，全部乘以该系数即可。而width_multiple表示BottleneckCSP模块的层缩放系数，将所有的BottleneckCSP模块的number系数乘上该参数就可以最终的层个数。可以发现通过这两个参数就可以实现不同大小不同复杂度的模型设计。比yolov4更加灵活。

纵观整个yolov5代码，和前yolo系列相比，特点应该是

**(1) 考虑了邻域的正样本anchor匹配策略，增加了正样本**

**(2) 通过灵活的配置参数，可以得到不同复杂度的模型**

**(3) 通过一些内置的超参优化策略，提升整体性能**

**(4) 和yolov4一样，都用了mosaic增强，提升小物体检测性能**

其他一些操作包括：

(1) 采用了最新版本的pytorch进行混合精度以及分布式训练

(2) warmup+cos lr学习率策略，对bias不进行权重衰减

(3) 采用了yolo系列中常用的梯度累积策略，增加batch size，并对输出head部分的bias进行特殊初始化；采用了类平衡采样策略


(4) 多尺度训练，但是写的非常粗暴，直接对dataloader输出的batch图片进行双线性插值

(5) 支持onnx格式导出

(6) 采用了模型权重指数滑动平均的ema策略(比赛常用策略)

# Day15

## yolov5的一些改进tricks总结

1.在yolov5上增加小目标检测层link

2.在yolov5上增加注意力机制
CBAM
SElayer
…
3.考虑在yolov5中加入旋转角度的目标检测机制。

4.结合BiPFN，将yolov5中的PANet层改为efficientDet中的BiFPN。

5.训练baseline，同时使用加权框融合WBF进行后处理/预处理。

6.AF-FPN替换金字塔模块。利用自适应注意力机制（AAM）和特征增强模块（FEM）来减少特征图生成过程中的信息丢失并增强表示能力的特征金字塔。将yolov5中原有的特征金字塔网络替换为AF-FPN。来解决模型大小和识别精度不兼容的问题，提高了识别多尺度目标的能力，并在识别速度和准确率之间做出有效的权衡。

7.从数据增强角度，用主动学习策略（active learning）来替换原有的mosaic augmentation。

**（**怎么融合的注意力机制：
主要是改三个地方，common.py yolo.py和对应预训练模型的yaml。
1.在common.py添加SElayer或CBAM的类。关于这个模块的定义去参考注意力机制的代码。大部分的注意力机制都是结合通道和空间去做文章。
2.在yolo.py开头的import环节添加common.py写好的注意力机制Module。
3.在对应预训练模型的yaml文件，backbone中嵌入你的注意力机制。**）**

8.yolov5结合BiFPN，现在的neck用的是PANet，在EfficientDet论文中提出了BiFPN结构，还有更加不错的性能，所以就尝试将yolov5中的PANet层改为BiFPN。

9.训练yolov5的baseline，同时使用加权框融合（WBF）进行后处理/预处理。

WBF是什么：

WBF已经成为优化目标检测的SOTA了。
如果你熟悉目标检测的工作原理，你可能知道总有一个主干CNN来提取特征。还有一个阶段是，生成区域建议（region proposal）–可能的建议框，或者是过滤已经提出的建议区域。这里的主要问题是，要么物体检测任务出现一物多框，要么生成的边框不够，最终导致平均精度较低的原因。目前其实已经提出了一些算法来解决这个问题。比如我们常见的NMS–非极大抑制。
但是其实，对于遮挡问题较为严重的检测任务，在一些目标密集的区域，可能包含多个标签，这意味着将出现一框多物的现象，如果使用非极大抑制NMS这类策略，它是通过iou来过滤框的，因此，很难确定一个较好的阈值，所以这类策略可能会删除有用的检测框。

另外还有soft-NMS，它试图通过一种更soft的方法来解决NMS的主要问题。它不会完全移除那些iou高于阈值的框，而是根据iou的值来降低它们的置信度分数。它是NMS的优化，相比于NMS会过滤掉过更少的框。

加权框融合（WBF）的工作原理与NMS不同。首先，它将所有的边界框按照置信度分数的递减顺序进行排序，然后生成一个可能的框来融合列表，并检查这些融合是否与原始框匹配。这里也会给定一个iou的阈值来判断匹配效果，它通过检查iou是否大于指定阈值来实现。

然后，通过一系列公式来调整坐标和框列表中所有框的置信度分数。新的置信度仅仅是它被融合的所有框的平均置信度。新坐标以类似的方式融合（平均），除了坐标是加权的，既然是加权的，意味着不是每个框在最终的融合的框中都具有相同的贡献。这个权重的值是由置信度来决定的，但较低的置信度可能表明预测错误。

除此之外，还有第四种方法，非最大加权融合，它的工作机制和WBF类似，但性能不如WBF，因为它不会改变框的置信度，而是使用iou值来衡量方框，而不是更精确的度量。其实表现也相当接近。

# Day 16
（yolov4的网络结构笔记）

## 1.CSPDarkNet53

CSPDarkNet53 骨干特征提取网络在 YOLOV3 的 DarkNet53网络 的基础上引入了 CSP结构。该结构增强了卷积神经网络的学习能力；移除了计算瓶颈；降低了显存的使用；加快了CNN网络的学习能力。

CSP结构图如下：图像输入经过一个3 * 3卷积的下采样层；然后输出特征图经过     1 * 1卷积分为两路分支，且卷积后的特征图的通道数为输入特征图通道数的一半。主干部分再通过1 * 1卷积调整通道数，经过若干个残差卷积块之后，再使用1 * 1卷积整合通道特征。最后将残差边和1* 1卷积输出特征图在通道维度上堆叠，再经过1*1卷积融合通道信息。

![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220719135614925.png)

模型的骨干就是由多个CSP结构组合而成，但是第一个CSP结构和其他的CSP结构不相同。以输入图像的shape为 [416,416,3] 为例。有如下两点不同：第一个CSP结构是先经过一个标准卷积块下采样，然后经过3* 3卷积提取特征，不改变通道数64；在主干卷积分支的残差块，先1* 1卷积下降通道数32，再3* 3卷积上升通道数64。

输入图像的shape为[416.416,3]，网络不断进行下采样来获得更高的语义信息，输出三个有效特征层，feat1的shape为 [52,52,256] 负责预测小目标，feat2的shape为[26,26,512] 负责预测中等目标，feat3的shape为[13,13,1024] 负责预测大目标
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/6fff7f87b86145409218597bca541128%20-%20%E5%89%AF%E6%9C%AC.png)

### Backbone的整体结构图：
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220719144823311.png)

## 2.SPP(空间金字塔)
主要功能是**将不同尺寸的输入转化为固定尺寸的输出**，使得神经网络在训练过程和推理过程都能摆脱对固定尺寸的依赖，避免了因为适应固定尺寸对原始图片进行裁剪（crop）或者变形（wrap）引起的信息丢失和位置信息扭曲。加强特征提取结构能在一定程度上解决多尺度的问题。如下图：
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220719152622396.png)

#  ***\*Panes\**** （路径聚合）

**PANet 将网络输出的有效特征层和SPP结构的输出进行特征融合**，它是由两个特征金字塔组成，**一个是将低层的语义信息向高层融合，另一个是将高层的语义信息向低层融合。**

首先，对SPP结构的输出p5进行卷积和上采样，对网络输出的26* 26* 512的特征图卷积，将两个结果在通道维度上堆叠，再经过5次卷积，输出特征图shape为26* 26* 256。然后将结果再进行卷积和上采样，网络输出的52* 52* 256的特征图经过1* 1卷积，两个特征图在通道维度上堆叠，完成特征金字塔的信息融合。

***\*Head*\***
YOLOHead 由一个3* 3卷积层和一个1* 1卷积层构成，3* 3卷积整合之前获得的所有特征信息，1* 1卷积获得三个有效特征层的输出结果。

其中1* 1卷积的通道数为 num_anchors(5+num_classes)。以 输出结果p3_output 为例，shape为 [512,512,num_anchors(5+num_classes)]，可理解为，将一张图片划分成 512*512 个网格，当某一个目标物体的中心点落在某网格中，该物体就需要该网格生成的预测框去预测。

每个网格预先设置了 num_anchors=3 个先验框，网络会对这3个先验框的位置进行调整，使其变成最终的预测框。此外，5+num_classes可以理解为4+1+num_classes。其中 4 代表先验框的调整参数(x, y, w, h)，调整已经设定好了的框的位置，调整后的结果是最后的预测框；1 代表先验框中是否包含目标物体，值越接近0代表不包含目标物体，越接近1代表包含目标物体；num_classes 代表目标物体的种类，VOC数据集中num_classes=20，它的值是目标物体属于某个类别的条件概率。

下图是借鉴别人的`YOLOv4`网络的详细结构图：

![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220719163535203.png)

# Day 17

# **基于Yolov4训练车辆检测模型**

**模型评估：**模型评估在一张32GTesla V100的GPU上通过'tools/eval.py'测试所有一部分验证集得到，单位是fps(图片数/秒), cuDNN版本是7.5，包括数据加载、网络前向执行和后处理, batch size是1。

**推理时间(fps)**: 推理时间是在一张32GTesla V100的GPU上通过'tools/eval.py'测试所有验证集得到，单位是fps(图片数/秒), cuDNN版本是7.5，包括数据加载、网络前向执行和后处理, batch size是1。

###### Baseline:

模型评估：

| 骨架网络   | 每张GPU图片个数 | 推理时间(fps) | mAP    |
| :--------- | --------------- | ------------- | ------ |
| CSPDarkner | 12              | 53.10         | 70.78% |

模型预测：

|      | 预测时间（ms） | 图片数量 |      |      |      |
| ---- | -------------- | -------- | ---- | ---- | ---- |
|      | 2398           | 1        |      |      |      |

预测结果：
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/%E4%B8%8B%E8%BD%BD.png)

【评价指标】：
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/loss.png)
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/loss_cls.png)
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/loss_iou.png)
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/loss_obj.png)
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/loss_wh.png)
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/loss_xy.png)
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/mAP.png)

# Day 18
### 1.SORT中的卡尔曼滤波

- SORT中将卡尔曼滤波器用于检测框运动的预测，那么描述一个检测框需要以下四个状态，即

（1）检测框中心的横坐标

（2）检测框中心的纵坐标

（3）检测框的大小（论文中叫做scale或者area）

（4）长宽比

- 以上四个状态可以描述一个检测框的基本信息，但是不能完全描述一个状态的运动状态信息，所以需要引入上述的状态的变化量信息（可以看作变化速度）来进行运动状态信息的描述。由于SORT假设一个物体在不同帧中检测框的长宽比不变，是个常数，所以变化量只考虑上面的（1）（2）（3），不考虑（4），即

（1）检测框中心的横坐标的变化速度

（2）检测框中心的纵坐标的变化速度

（3）检测框的大小（论文中叫做scale或者area）的变化速度

- 所以SORT中共使用了7个参数，用来描述检测框的状态。将帧间的位移假设为线性匀速模型，所以每个目标的状态综合了上述提到的7个信状态值。

## 2 DeepSORT

### 2.1 简介

- DeepSORT是SORT算法中的升级版本，该算法中使用到了Deep Learning网络。为解决SORT算法受遮挡等情况影响较大，会有大量的ID切换的问题

### 2.2 解决办法

- 利用到前面已经检测到的物体的外观特征，当物体遮挡结束，能够利用之前保存的外观特征分配该物体受遮挡前的ID编号，降低ID切换。
- 采用了一个简单的CNN，结构如下图，来提取被检测物体的外观特征，在每帧检测+追踪后，进行一次物体外观特征的提取并保存。后面每执行一步时，都要执行一次当前帧被检测物体外观特征与之前存储的外观特征的相似度计算，这个相似度将作为一个重要的判别依据（不是唯一的，因为作者说是将运动特征与外观特征结合作为判别依据，这个运动特征就是SORT中卡尔曼滤波做的事）。

![img](https://ai-studio-static-online.cdn.bcebos.com/eae7f85ca57a4673bdc95ed9211dbaa07cc694cecbe64bca81d72bf2aea2c1ea)
### 2.3 DeepSORT中的卡尔曼滤波

- SORT中的卡尔曼滤波算法使用的状态是一个7维的向量。
- 在DeepSORT中，使用的状态是一个8维的向量
- 相较于SORT中的状态，多了一个长宽比（aspect ratio）的变化率。实际过程中，随着镜头移动或者物体与相机的相对运动，物体的长宽比也是会发生变化的。
- 同时，DeepSORT对追踪的初始化、新生与消失进行了设定。

（1）初始化：如果一个检测没有和之前记录的track相关联，那么从该检测开始，初始化一个新的目标（并不是新生）

（2）新生：如果一个目标被初始化后，且在前三帧中均被正常的捕捉和关联成功，那么该物体产生一个新的track，否则将被删除。

（3）消失：如果超过了设定的最大保存时间（原文中叫做predefined maximum age）没有被关联到的话，那么说明这个物体离开了视频画面，该物体的信息（记录的外观特征和行为特征）将会被删除。

### 2.3 DeepSORT中的分配问题

- 解决分配问题使用的是匈牙利算法（仅使用运动特征计算代价矩阵），该算法解决了由滤波算法预测的位置与检测出来的位置间的匹配。DeepSORT中，作者结合了外观特征（由小型CNN提取的128维向量）和运动特征（卡尔曼滤波预测的结果）来计算代价矩阵，从而根据该代价矩阵使用匈牙利算法进行目标的匹配。

## 3 JDE

------
### 3.1 简介

- Jointly learns the Detector and Embedding model (JDE)，是一类追踪算法的总称。该范式首先通过检测器（detector）检测出画面中物体所在的检测框，然后根据物体检测框移动的规律（运动特征）和检测框中物体的外观特征（通常通过一个ReID网络抽取一个低维的向量，叫做embedding向量）来进行前后帧同一物体的匹配，从而实现多目标追踪。
- 若将MOT分为了两步，即

（1）物体检测

（2）特征提取与物体关联

- 该类方法检测与特征提取是分开的，所以又被称为SDE（Separate Detection and Embedding），SDE存在的最大缺点就是速度慢，因为将物体检测和（外观）特征提取分开，检测速度自然就下去了。

### 3.2 JDE的网络结构和损失函数

- 该方法是基于One-stage检测器学习到物体的embedding的（代码中采用的是经典的YOLO V3模型）。那么JDE范式就应该在检测器的输出（head），多输出一个分支用来学习物体的embedding的。结构如下图:


![img](https://ai-studio-static-online.cdn.bcebos.com/83d488f8d2d0445dbd423161e31553c4b4c5f180be92493d9ba18dd736b00c4d)
FPN网络产出Prediction head，在Prediction head中多出来了一个分支用于输出embedding。然后使用一个多任务学习（multi-task learning）的思路设置损失函数。

### 3.3 embedding操作方法

- 对embedding进行转化，转化为足够强的语义信息，也就是这个embedding到底属于哪个track ID的物体，那么这种就需要借鉴物体分类的思路了（将每个track ID当作一个类别），所以作者引入了全连接层将embedding信息转化为track ID分类信息，如下图。
![img](https://ai-studio-static-online.cdn.bcebos.com/47d206ea200549e3905f45da1b439a84be22d3cfdcb24e5cb869e94eafaf7883)

### 3.4 小结

JDE是一个网络同时输出图像画面中的检测框位置和检测框内物体的embedding，加速MOT的速度。JDE只是同时输出了检测框和embedding信息。后面还是要通过卡尔曼滤波和匈牙利算法进行目标的匹配。

## 4 FairMOT

### 4.1 简介

- 其实FairMOT属于JDE（Jointly learns the Detector and Embedding model ）的一种。实验证明了现有的JDE方法存在一些不足，FairMOT根据这些不足进行了相关的改进

### 4.2 JDE方法存在的问题

- Anchor-Based 的检测器不适用JDE追踪模式，如下图
- ![img](https://ai-studio-static-online.cdn.bcebos.com/8127d039721841f483a8efd75eaf2359c0f8399485c046daabd256c1bceca30a)

（1）一个物体可能被多个anchor负责并进行检测，这会导致严重的网络模糊性。

（2）实际物体的中心可能与负责对该物体进行检测的anchor中心有偏差。

- 缺少多层特征融合

Re-ID信息不能仅仅包含高层网络中的语义信息，也要适度包含低层网络中的的颜色，纹理的信息。

- Re-ID存储维度选择

MOT的一些benchmarks中并没有那么像Re-ID那么多的数据，维度设置大了容易过拟合，低维度的特征更适用于JDE这种MOT方法。

### 4.3 FairMOT解决方法

- 针对Anchor-Based 的检测器不适用JDE追踪模式问题，使用了Anchor-Free目标检测范式来代替。anchor-free的方法就是一种基于关键点检测的方法。目前最常见的就是检测物体的中心点。本文中正是采用了基于中心点检测的anchor-free方法，该方法可以有效解决上述讨论的问题。
- 针对多层特征融合问题，选择DLA( Deep Layer Aggregation)的网络进行特征提取，这个网络的最大特点就是多层融合，结构如下图。

![img](https://ai-studio-static-online.cdn.bcebos.com/b4bfd97960a84f83a4c45aec156b2786b21eddfaa1964f71a3777571672c6b48)

- Encoder-decoder网络提取的（stride=4）高分辨率特征图将被作为四个分支的特征图。其中三个被用来检测物体（Detection），一个被用来输出物体的Re-ID信息(Re-ID)。
- 每个分支都被称为一个head分支。每个head除了最后输出通道维度的不同，其组成都类似，每个head由一个3x3卷积层后面接一个1x1卷积层实现的。

### 4.4 FairMOT小结

- FairMOT获得物体的位置和Re-ID信息后，配合卡尔曼滤波求解其代价矩阵（cost matrix），然后利用匈牙利算法进行匹配，FairMOT就结束了。

------


# Day20

1️⃣ 	***改进【一】：*** 将Mish激活函数换为Hardswish激活函数

yolov4的主干网络使用的是Mish激活函数，是一种平滑的非单调的激活函数。

![image-20220720141809626](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220720141809626.png)

Mish激活函数的复杂度较高，计算的同时要引入一个softplus()激活层和另外一个tanh激活函数，从而导致计算速度较慢。

Hardswish激活函数。相较于swish函数，具有数值稳定性好，计算速度快等优点。其数学表达式见公式：

![image-20220720141056224](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image.png)
hardswish激活函数是对swish激活函数的改进，由公式可见hardswish激活函数可以实现为分段功能，以减少内存访问次数，从而大大降低了等待时间成本。

![image-20220720165925991](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220720165925991.png))

改动前：

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarkner+Mish | PANet+L-relul | 53.10 | 69.78% |

改动后模型评估：

| 骨架网络           | Neck          | fps   | mAP    |
| :----------------- | ------------- | ----- | ------ |
| CSPDarkner+H-swish | PANet+L-relul | 52.06 | 69.76% |

改动前：

![lossa](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/lossa.png)

改动后：

![lossaa](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/lossaa.png)

总结：对比前后两个loss变化图可以看出，改动后训练时间缩短了10分钟左右，loss下降的更快一些；从预测结果来看，预测帧率变化不大，但是精度下变化不大；综合来看，用Hardswish激活函数替换Mish函数效果并不十分明显。

# Day21

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
![lossaaaaa](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/lossaaaaa.png)

![mAPaa](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/mAPaa.png)

总结：可以看到，当我们把Neck的激活函数也换为H-swish后，loss函数的收敛值是比较高的，而且精度也有所下降，可见在yolov4的CSPDarkner、PANet网络中，Mish和L-relul激活函数还是比较好的。

# Day22

3️⃣	***改进【三】：*** 增加SE注意力机制

SENet 架构论文连接：https://arxiv.org/abs/1709.01507

通过卷积神经网络可以融合通道信息来构建信息特征，通过增加SENet架构可以有效地增加卷积网络的表达能力，“SE注意力机制通过显式建模通道之间的相互依赖关系来自适应地重新校准通道特征响应”，从而能提升网络性能。

![image-20220721221546240](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%Photo/image-20220721221546240.png)

![image-20220721221604628](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220721221546240.png)

![image-20220721221625993](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220721221625993.png)
在箭头表明的地方增加了SE注意力机制：

![image-20220722110231157](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220722110231157.png)

改动前：

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarkner+Mish | PANet+L-relul | 53.10 | 69.78% |

增加SE后模型评估：

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 33.46 | 70.57% |

loss变化图像：

![lossaaaa](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/lossaaaa.png)

总结：可以看到，增加SE注意力机后模型的训练速度慢了一个小时左右，但是模型评估的精度比Baseline增加了一个多点（此时我已经把激活函数改回了Mish），但推理的FPS有所下降，可见在卷积网络中增加注意力机制的办法确实可以提升网络性能。（两次修改后预测图片的结果都差不多，所以就没有贴，可能是图片找到不太好，下次换一张图来预测）

# Day24

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

# Day25

5️⃣  ***改进【五】：***  修改学习率

学习率可以说是**模型训练最为重要的超参数**。 通常情况下，一个或者一组优秀的学习率既能加速模型的训练，又能得到一个较优甚至最优的精度。 过大或者过小的学习率会直接影响到模型的收敛。因为学习率是官方设置好的，本来没想到要修改学习率，由于改了几个点模型的精度都没有明显的提升，所以我尝试在paddle发布的原版本的基础上修改了'base_lr'=0.001(原来是0.0001)，将'milestones'的范围改为（20000，40000），总iter为60000，将学习率变化的范围分为1:1:1

![image-20220726231557008](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220726231557008.png)

修改后训练完的模型精度竟然达到了85.05%

| 骨架网络        | Neck          | fps   | mAP    |
| :-------------- | ------------- | ----- | ------ |
| CSPDarknet+Mish | PANet+L-relul | 33.91 | 85.05% |

（加SE注意力机制、换成aspp模块）loss变化及精度指标：

![lss](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/lss.png)



![mAaP](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/mAaP.png)

总结： 通过适当增加模块，增大感受野，提高特征提取率，修改学习率后，模型精度大幅超过了Baseline，loss的收敛值接近了3，但是最低loss值在大约40000iter左右，之后又有所升高。map达到了85.05%的高精度。可见在修改模型后适当调整学习率会有更好的效果。




