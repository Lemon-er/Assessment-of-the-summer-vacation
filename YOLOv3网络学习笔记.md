YOLOv3网络学习笔记

v3是anchor based算法，预先在图片上生成很多先验框，然后通过神经网络去判断放的框内有没有我们想要的特征物体，如果有，就用神经网络对先验框的中心和长和宽进行调整，最终输出物体的长和宽。

**YOLOv3**的网络有**三个输出**，对于输入为(3,416,416)的图片，通过这个网络之后会输出(75,13,13)，(75,26,26)，(75,52,52)，分别对应有13×13、26×26、52×52的不同大小网格的三张图片，图像分成了**多个网格**，每个网格上都会放置好**3个先验框**，先验框的**长宽是一开始就固定的**，**13×13的网格用于检测大物体，26×26的网格用于检测中等物体，52×52的网格用于检测小物体**。一共有**13×13×3+26×26×3+52×52×3**个框。**物体中心点落在哪个框内，这个框就负责识别这个物体。**

**75的含义是3×(num_class+4+1)**，这是对应**voc**数据集，有**20**个类别。**20指的是20个类别的概率**，**4指对先验框的中心点和长宽的修正量，1指这个框内有没有物体。**因为每个格子都有**3个先验框**，所以再乘3。如果使用**coco数据集**，有80个类别，那么对应的维度就是**3×(80+4+1)=255**。

## 整体网络：

分为三部分，分别为backbone（主体）,neck（脖子）和head（头部）。

![image-20220702145221218](https://github.com/nongfulv2/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220702145221218.png)

## backbone部分：

为DarkNet53，称为特征提取网络，和分类网络基本一致，这部分是通用的（可以做分类任务）

总体形式如下：

![image-20220702150614059](https://github.com/nongfulv2/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220702150614059.png)

​		先通过卷积层把输入通道数扩充到32，然后接着很多个Residual Block,中间穿插很多向下采样（Conv）。Residual Block包含两个卷积归一化层，第一个卷积归一化层就是通过1 * 1的卷积把输入通道数降到一半，然后经过归一化和激活函数；第二个卷积层主要用来提取特征，大小为3 *3，再将输出矩阵的通道数与输入通道数改为一致，然后经过归一化和激活函数后将输入与输出相加（残差结构）得到输出结果，此结果也是下一个卷积归一化的输入。可见DarkNet53借鉴了残差结构，残差结构用于深度比较深的网络可以防止梯度爆炸或梯度消失。

​		YOLO v3有三个输出，决定了head有三个输出，则DarkNet53也得有三个输出（上图中的Output Feature），这三个输出结果会输入到neck中。



## Neck部分：

使用的是**FPN**，这一部分也叫**特征金字塔**，它的作用是将**多尺度的出入进行特征融合**。

### 整体结构：

**backbone**部分输出的shape分别为（13，13，1024），（26，26，512），（52，52，256）。将这三个输出分别输入到FPN中，（13，13，1024）这一个输入，**经过5次卷积后**，输出（13，13，512），然后**兵分两路**，一路传入到**head**中，一路再经过**一个卷积和上采样**，得到（26，26，256），将这个输出和**backbone**的第2个输出也就是（26，26，512）**堆叠**（**concat**）,得到（26，26，768）。

（堆叠：大小不变，通道数相加）

卷积池化使得图像**大小越来越小**，这个过程叫**下采样**。**上采样就是通过插值的方法，扩充图像大小**，就像这边的（13，13）变成（26，26）。

**concat**操作是把两个矩阵**堆叠**到一起，里面的数不变，只是单纯的堆叠，**shape等于两个相加**。残差结构中的**add**区分开，相加操作需要**两个输入shape一致，里面的数也会相加**。

后续部分其实就是第一种操作的重复。堆叠后的矩阵，经过5次卷积，再兵分两路，一路输入head，一路经过卷积上采样，和backbone第一个输出堆叠。最后通过五次卷积，输入到head中。

解释图:

![image-20220702192347205](https://github.com/nongfulv2/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220702192347205.png)

获得到网络输出的三个矩阵后，并不是**直接获得了我们最后的预测框**。我们之前说过对于voc数据集，75=3×（20+4+1），其中20和1都是和分类相关的，对于这个4，也就是对先验框的4个调整的参数，通过调整后也就输出了最后的预测框。

先验框是**固定不变的**，每个特征图，每个图的**每个格子有3个先验框**，所以预先准备**9个大小的先验框**。

anchors大小：[116,90],[156,198],[373,326],[30,61],[62,45],[59,119],[10,13],[16,30],[33,23]，这些大小是**相对于416×416的尺寸**，我们最后三个输出的大小为13×13，26×26，52×52，所以要进行**相应的缩放**。

我们以13×13的输出为例，原本416×416大小变成13×13，相当于缩小了**32倍**，也就是说**原图32×32个小方块对应于最后输出的1×1的像素点**。anchors[116,90],[156,198],[373,326]相应地长宽都应该**除以32**，这就是**13×13每个点上的三个先验框**。

![image-20220702195440234](https://github.com/nongfulv2/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220702195440234.png)

**粉红色**的就是对应到13×13上的**先验框**，它是我们一开始自己就确定的，显然是**不正确**，需要模型对它调整。

先验框怎么摆放的呢，它的中心就是落在**13×13的交点**上，长宽就是除以32的结果。先验框坐标记为（cx,cy,pw,ph），模型输出的4为（tx,ty,tw,th）,调整的公式如上图所示，中心点取**sigmoid激活函数**，**sigmoid函数范围是0-1**，也就是中心点的调整范围永远在**右下角的框内**，这也就是我们说的，**物体的中心落在哪个格子里，就有哪个框负责预测**。

**长宽取exp后相乘**。这就得到了在13×13尺寸图上的预测框，然后**再乘以32缩放回来就得到了最后的预测框**。
