
Ai studio 使用教程：

Ai studio地址：[飞桨AI Studio - 人工智能学习实训社区 (baidu.com)](https://aistudio.baidu.com/aistudio/index)

![image-20220704132941869](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220704132941869.png)

使用 Ai studio 的算力需要相应的算力卡，进行登陆注册后可领取”新手礼包“获得算力卡，而且每天使用Ai studio的GPU也会送8小时的算力卡。

在Ai studio上运行项目首先要将数据集(包括项目的全套代码)打包上传：

![image-20220704133449666](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220704133449666.png)

![image-20220704133711501](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220704133512503.png)

![image-20220704133735350](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220704133849996.png)

下一步创建项目-->选择类型（Notebook）-->环境配置（Ai studio经典版）-->添加项目描述即可。

创建好项目以后，点击启动环境即可

![image-20220704134121977](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220704134121977.png)

![image-20220704135723008](https://github.com/Lemon-er/Assessment-of-the-summer-vacation/blob/main/My%20Photo/image-20220704135723008.png)

根据需要选择相应的GPU大小即可，然后进入环境，首先将上传的数据集压缩包解压，数据集放在“data”文件夹下，找到压缩包点右边的三个小点复制解压命令，在终端中运行命令即可。

然后‘cd’,进入解解压后文件夹（也就是在本地打包压缩的文件夹）的目录下，进行模型训练：

模型训练、断点训练、模型评估、模型导出、模型预测有以下相应的命令:

（根据文件目录修改就行，命令结构基本一致）

### 模型训练

```
python tools/train.py -c configs/yolov3/yolov3_darknet53_270e_voc.yml --use_vdl=True --eval
```

### 断点训练

```
python tools/train.py -c configs/yolov3/yolov3_darknet53_270e_voc.yml -r output/yolov3_darknet53_270e_voc/100
```

### 模型评估

```
python tools/eval.py -c configs/yolov3/yolov3_darknet53_270e_voc.yml -o weights=output/yolov3_darknet53_270e_voc/best_model
```

### 模型导出

```
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_voc.yml --output_dir=./inference_model -o weights=output/yolov3_darknet53_270e_voc/best_model
```

### 模型预测

```
python deploy/python/infer.py --model_dir=./inference_model/yolov3_darknet53_270e_voc --image_file=./street.jpg --device=GPU --threshold=0.2
```

### 安装包

```pip install pycocotools lap motmetrics```

paddlepaddle 可视化：
输出值：loss偏差逐渐降低，acc1的解释：经过模型训练后一般会输出由n个概率值组成的列表（n为标签的数量）例如本次训练有4个标签，可能输出的结果为[0.6 0.2 0.1 0.1],acc1就是该标签的第一个概率值对应的标签是实际的标签就认为是正确的，acc4就是n个概率值降序排列后，前面4个概率值对应的标签包含实际的标签就认为是正确的。

model.train()中的"use_vdl=Ture"的设置可以在模型输出“output”文件夹生成“vdl_log”文件夹，作为模型训练的可视化工具。具体操作：

在cmd命令行的output所在目录下执行激活“paddle_env”，然后输入命令行

`visualdl --logdir output/mobilenetv3_small --port 8001`

（“output/mobilenetv3_small”为“vdl_log”文件夹所在的两个上级目录）

cmd输出有一个网址，复制到浏览器中打开就有训练时相关参数的变化过程。
