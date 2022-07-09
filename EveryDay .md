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
|         |                |          |      |      |      |
第一次预测：

street:

![image-20220708203852435](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/street.png)
street1:
![img](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/tempsnip1.png)
