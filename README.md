# Semi-Supervised-Learning

# 目录
- [环境配置](#环境配置)  
- [文件目录](#代码框架)  
- [运行](#运行)  
	- [训练](#训练)  
	- [测试](#测试)  
- [常用参数](#参数设置)  
- [专属参数设置](#专属参数设置)  
- [数据集说明](#数据集说明)  
	- [研究点一（sgrw）数据集说明](#数据集说明)
	- [研究点二（sgpw）数据集说明](#数据集说明)
	- [算法链接](#算法链接)
- [各算法命令行](#运行)
	- [算法](#不同算法)
    	- [训练](#训练)
    	- [测试](#测试)
	- [研究点一命令行](#运行)
	- [研究点二命令行](#运行)
- [可视化](#运行)


## 配置环境

```
* python >= 3.6  
* cuda >= 11.0   
grad-cam = 1.2.9  
```

## 文件目录

```
┌── data
│  ├── /data/
│  │  ├── /train/
│  │  │  ├── /cj1/
│  │  │  ├── /cj2/
│  │  │  │  ├── /normal_with_cam_part_scenario/
│  │  │  │  │  ├── /gyy/
│  │  │  │  │  │  ├── test_gyy_bag_02_0_1.jpg
│  │  │  │  │  │  ├── test_gyy_bag_02_0_2.jpg
│  │  │  │  │  │  └── ...
│  │  │  │  │  ├── /jhr/
│  │  │  │  │  └── ...
│  │  │  │  └── /part_part_scenario_cam/
│  │  │  └── /cj3/
│  │  └── /test/
│  ├── /data2/
│  │  ├── /径向走-随机走身份识别实测数据集/
│  │  │  ├── /all_random_constant/
│  │  │  │  ├── /constant/
│  │  │  │  │  ├── /gyy/
│  │  │  │  │  │  ├── gyy_constant_01_407.jpg
│  │  │  │  │  │  ├── gyy_constant_0._408.jpg
│  │  │  │  │  │  └── ...
│  │  │  │  │  ├── /jhr/
│  │  │  │  │  └── ...
│  │  │  │  ├── /constant_5/
│  │  │  │  ├── ...
│  │  │  │  ├── /random/
│  │  │  │  ├── /random_5/
│  │  │  │  ├── ...
│  │  │  │  └── /test/
│  │  │  └── ...
│  │  ├── /全向（0°-330°）身份识别实测数据集/
│  │  │  ├── /train/
│  │  │  │  ├── /other/
│  │  │  │  │  ├── /ljh/
│  │  │  │  │  │  ├── ljh_30_walk_01_1.jpg
│  │  │  │  │  │  ├── ljh_30_walk_01_2.jpg
│  │  │  │  │  │  └── ...
│  │  │  │  │  ├── /scy/
│  │  │  │  │  └── ...
│  │  │  │  ├── /other_1_10/
│  │  │  │  └── ...
│  │  │  ├── /test/
│  │  │  │  ├── /test/
│  │  │  │  └── /test_jiaodu/
│  │  │  └── ...
│  │  ├── /全向（0°-350°）身份识别实测数据集/
│  │  │  ├── /train/
│  │  │  │  ├── /labeled/
│  │  │  │  │  ├── /box/
│  │  │  │  │  ├── /crawl/
│  │  │  │  │  └── ...
│  │  │  │  └── /unlabeled/
│  │  │  └── /test/
│  │  └── ...
│  └── ...
├── Semi-Supervised_Learning (本级目录)
│  ├── README.md
│  ├── vis_readme.txt
│  ├── /Vis/
│  │  ├── attentionmap.py
│  │  ├── featuremap.py
│  │  ├── filtermap.py
│  │  ├── __init__.py
│  │  ├── logger.py
│  │  ├── loggin_save_print.py
│  │  ├── monitor.py
│  │  ├── netinfo.py
│  │  ├── processed_image.py
│  │  ├── save_embedding.py
│  │  └── valueepochimg.py
│  ├── /cls_dataset/
│  │  ├── CIFAR10.py
│  │  ├── CIFAR100.py
│  │  ├── ImageFolder.py
│  │  ├── leidaFolder.py
│  │  └── MNIST.py
│  ├── /libs/
│  │  ├── /openset_metric/
│  │  │  └──openset_metric.py
│  │  ├── Augment.py
│  │  ├── base_model_d.py
│  │  ├── image.py
│  │  ├── leida_image.py
│  │  ├── pre_data.py
│  │  ├── metric.py
│  │  ├── opt.py
│  │  └── Visualizer.py
│  ├── /multi_class/
│  │  ├── /meanteacher/
│  │  │  ├──model.py
│  │  │  └──meanteacher_network.py
│  │  ├── /adamatch/
│  │  │  ├──model.py
│  │  │  └──adamatch_network.py
│  │  ├── /fixmatch/
│  │  │  ├──model.py
│  │  │  └──fixmatch_network.py
│  │  ├── /pl/
│  │  │  ├──model.py
│  │  │  └──pl_network.py
│  │  ├── /vat/
│  │  │  ├──model.py
│  │  │  └──vat_network.py
│  │  ├── /uda/
│  │  │  ├──model.py
│  │  │  └──uda_network.py
│  │  ├── /mixmatch/
│  │  │  ├──model.py
│  │  │  └──mixmatch_network.py
│  │  ├── /sgrw/
│  │  │  ├──model.py
│  │  │  └──ours_network.py
│  │  ├── /sgwp/
│  │  │  ├──model.py
│  │  │  ├──regularizer.py
│  │  │  └──sgwp_network.py
│  │  └── /remixmatch/
│  │     ├──model.py
│  │     └──remixmatch_network.py
│  └── main.py
└── output
```
注：运行时，请将NAS上 “研究点一数据/data” 或“研究点一数据/data”文件夹直接下载到算法工程文件夹中，其中data和libs、cls_dataset、Vis等同级别。
```
data数据集：即研究点一数据集，该数据集为9人伪装数据集（在我的论文中我将该数据集称为9人身份实测数据集）。该数据集在NAS上已经整理好，data分为train 和test。  
运行时，请将NAS上 “研究点一数据/data” 文件夹直接下载到算法工程文件夹中，其中data和libs、cls_dataset、Vis等同级别。
每组文件夹的调用都在算法的libs/leida_image.py中对cj==1 cj==2 cj==3进行if-else if判断来选择(这样不太好，若能改进就好了)，opt.py中设定默认的cj为2，也就是主实验。cj1对应这论文中的case(i)。cj3对应论文中的case(ii)
```
```
data2数据集：即研究点二数据集。研究点二包含三个数据集：①随机行走数据集②全向身份识别数据集③全向动作检测数据集，每个数据集都是多分类，各有六个类。三个数据集在NAS上已经整理好，data分为train和test。
```
## 运行
```
设置数据集目录后，代码可以通过以下脚本运行。

注：训练周期包含epoch eval_step total_steps，其中step与iter意义相同，且一般是通过设定iter的数目来设定训练次数。
```
### 训练
```sh
python main.py <--ssl_model model_name> <--args1 args1_value> <--args2 args2_value>
例：python main.py --ssl_model sgwp --cls_dataset ImageFolder --n_classes 9 --mode train
```
### 测试
```
python main.py <--ssl_model model_name> --phase test <--args1 args1_value> <--args2 args2_value>
例：python main.py --ssl_model sgwp --cls_dataset ImageFolder --n_classes 9 --mode test
```
## 常用参数
```python
--cls_type # multi_class
--ssl_model # 可选 ['vat','pl','mixmatch', 'uda', 'fixmatch','remixmatch','meanteacher','wideres']
--mode # 默认为 "train"
--cls_dataroot # 数据集所在根目录,默认为"../data"
--cls_dataset, # 数据集名称，
--cls_imageSize 32 # 输入图片尺寸
--cls_imageSize2 -1
--nc 3 # 通道数
--cls_batchsize # 批处理大小，默认64
--metric ACC  # 待计算指标
--metric_pos_label None # 类别个数范围内（0~n-1)内的某个值，默认为9
--metric_average micro # sklearn计运算某些指标(precision,recall,F1_score)时需要用到的平均方式,默认为'micro'
--aug_methods nargs +
--outf # 输出路径，默认'../output'
--control_save_end 1 # 在终端保存权重,默认为0(不保存)
--control_print True # 在终端打印结果,默认为真
--control_save True # 将结果保存成文件,默认为真
--load_weights # 加载权重,触发时为真
--gray_image # 将输入图片转成灰度图,触发时为真
--channel # 使用或不使用通道转换所有数据，默认为0

--wresnet_k 2 # resnet的宽度参数
--wresnet_n 28 # resnet深度
--use_ema true # 是否使用ema模块
--ema_alpha 0.999 # ema模块的延迟率

--n_classes # 数据集中的类数
--n_labeled 250 # 训练的标记样本数
--batch_size 8 # 训练标记样本的批量大小
--mu 7 # 未标记样本的train批量系数
--cj 2 # 选择的数据集文件夹
--root ../data # 数据集根目录

--lr 0.03 # 训练的学习率
--weight_decay 5e-4 # 权重衰减
--momentum 0.9 #动量
--seed 11 # 为随机行为提供种子，如果为否定则无种子”

--num_workers 4 # number of workers
--total_steps 1024*1024 # 总steps数
--eval_step 1024 # 总eval数
--expand_labels # 展开标签以适应评估步骤
--local_rank -1 # 对于分布式培训：local_rank
--warmup 0 # warmup epochs（基于未标记的数据）
--no_progress # “不使用进度条
--test_interval 1 # 训练多少epoch进行一次测试

--control_monitor 0 # 使用visdom监视loss和metric变化,默认为0(不开启监视)
--control_save_img_type lossepoch metricepoch t-SNE # 结果可视化,可加['featuremap', 'filter', 'attentionmap','confusionmap', 'processed', 'valueepoch','t-SNE']
--visdom_port 8097 # 输入visdom服务器的端口号
--embedding_dim 2 # 降维后的（t_SNE）特征维数.默认为2
--tsne_init pca # （t_SNE）特征分布初始化方式,默认为"pca"
--target_category None # target_category也可以是一个整数，或批处理中每个图像的不同整数列表,默认为None
```
## 专属参数设置
### VAT 专属参数
这些参数代表VAT中的loss权重以及VAT中特殊的perturbation size值
```python
--entmin_weight 0.06 #熵最小化权重
--vat_eps 6 # VAT扰动大小
```
### sgrw专属参数
这些参数代表新添加的之路中的ema_mask变化的epoch与变化的范围并且在变化后在剩余的epoch中一直保持ema_mask_end_w的值
```python
--ema_mask_init_ep 0 # ema mask 的初始更改epoch
--ema_mask_end_ep 50  # ema mask 的最终更改epoch
--ema_mask_init_w 1 # ema mask 的初始权重
--ema_mask_end_w 3.0 # ema mask 的初始权重
```
### ReMixMatch专属参数
相较于MixMatch，Remixmatch中额外添加了连个损失项，所以也加入了两个独有的损失的权重。
```python
--lam_us 0.5 # Lus的权重
--lam_rot 1.5 # Rotloss的权重
```
## 数据说明
```
半监督学习的输入数据包括有标签数据与无标签数据，因此准备两个dataloader进行训练，并在其中设定不同的数据增强方法(随算法改变而改变)
各个算法model.py中都添加了对数据增强的设定，包含对有标签数据和无标签数据的预处理。其涉及的参数包括
"--labeled_aug":表示对有标签数据进行增强方式的选择，包括weak，strong，normalize，nothing
"--un_img1_weak"：表示对无标签图像1进行增强方式的选择，若为True则是weak，否则nothing
"--un_img2_strong"：表示对无标签图像2进行增强方式的选择，若为True则是strong，否则weak

在半监督学习中存在强增强与弱增强的设定，weak弱增强则是简单的镜像、翻转的结合，而强增强则在弱增强的基础上包含额外的增强方式，比如RandAugment、CTAugment、AutoAugment、Mixup等，此库中的strong一般指RandAugment，因为该方法比较简洁且效果更佳。
```
### 研究点一（sgrw）数据集说明
```
train中包含三个压缩文件，cj1，cj2，cj3代表不同的有标签/无标签数据的配比情况，将它们全部解压到train内，会得到3组有标/无标文件夹，共6个文件夹。test中只包含一个压缩包，开压缩即可。
所有的微多普勒频谱图都是有标签数据，但是在训练过程中会忽略许多数据的标签，将他们当作无标签数据看待。
终端中添加输入参数--cj 1就是利用cj 1进行实验。训练数据有三组，测试数据中只有一组。Eg.
FixMatch算法
python main.py --ssl_model fixmatch --cls_dataset ImageFolder --n_classes 9 --cj 1
python main.py --ssl_model fixmatch --cls_dataset ImageFolder --n_classes 9 --cj 2
python main.py --ssl_model fixmatch --cls_dataset ImageFolder --n_classes 9 --cj 3
注意：.sh文件中都未添加cj的设置，也就是默认都是2，若要尝试1或3的情形，则自行复制添加即可。
```
### 研究点二（sgwp）数据集说明
```

所有的微多普勒频谱图都是有标签数据，但是在训练过程中会忽略许多数据的标签，将他们当作无标签数据看待。
注：每个数据集train中包含压缩文件，将压缩文件中的文件夹或者部分文件夹解压到train内，会得到多组有标/无标文件夹。
每个文件夹都有自己的属性名称，代表着原本的无标签数据中所占有的有标签样本的比例。例如：constant代表径向行走，random代表随机行走，那么constant_500就代表从random中每个类随机取出1/500个(占比0.2%)样本放入到constant文件夹中作为有标签样本，有标签样本数目增加了，无标签样本数目减少了，对用的无标签样本文件夹则为random_500，以此类推。
test中直接解压缩即可。
与研究点一不同，这里的数据集可能会很多中有标/无标的配对情况，这种情况可以在libs/leida_image.py中自行去设定对应的cj值[在if else处设置就好]， 

例如(随机行走数据集，径向行走[constant]作为有标，随机行走[random]作为无标):
elseif self.args.cj==200:
	self.train_labeled_dataset_root=’./train/constant_200’
	self.train_unlabeled_dataset_root=’./train/random_200’
	self.test_dataset_root_all = ‘./test’ #对应的名称
注：注意中英文的引号

由于算法本身结构复杂，程序运行需要很长的时间。
同样的可以缩减对应的iter次数或epoch次数进行选择性调整。
```


### 算法链接
```
算法链接:[Mean Teacher](https://dl.acm.org/doi/pdf/10.5555/3294771.3294885) | [FixMatch](https://arxiv.org/ftp/arxiv/papers/2001/2001.07685.pdf) | [MixMatch](https://arxiv.org/abs/1905.02249) | [Pseudo-Labeling](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.664.3543&rep=rep1&type=pdf) | [ReMixMatch](https://arxiv.org/pdf/1911.09785.pdf) | [UDA](https://arxiv.org/pdf/1904.12848.pdf) | [VAT](https://ieeexploreieee.53yu.com/abstract/document/8417973) 
```
## 各算法命令行
### Mean Teacher

#### train model
```bash
python main.py --ssl_model meanteacher --weight_decay 5e-4 --lr 0.0003 --cls_dataset ImageFolder --n_classes 6 --mu 1 --total_steps 30000 --eval_step 1000 
```
#### test
```bash
python main.py --ssl_model meanteacher --cls_dataset ImageFolder --n_classes 6 --mode test 
```

### MixMatch

#### train model
```bash
python main.py --ssl_model mixmatch  --cls_dataset ImageFolder --n_classes 9 --mu 1
```
#### test
```bash
python main.py --ssl_model mixmatch --cls_dataset ImageFolder --n_classes 9 --mode test
```
### Pseudo-Label

#### train model

```bash
python main.py --ssl_model pl --mu 1 --cls_dataset ImageFolder --n_classes 6
```
#### test
```bash
python main.py --ssl_model pl --cls_dataset ImageFolder --n_classes 6 --mode test
```
### VAT

#### train model

```bash
python main.py --ssl_model vat --mu 1 --cls_dataset ImageFolder --n_classes 6
```
#### test
```bash
python main.py --ssl_model vat --cls_dataset ImageFolder --n_classes 6 --mode test
```

### FixMatch

#### train model

```bash
python main.py --ssl_model fixmatch  --cls_dataset ImageFolder --n_classes 6
```
#### test
```bash
python main.py --ssl_model fixmatch --cls_dataset ImageFolder --n_classes 6 --mode test
```
### UDA

#### train model
```bash
python main.py --ssl_model uda  --cls_dataset ImageFolder --n_classes 6  --mu 1 --lr 0.1
```
#### test
```bash
python main.py --ssl_model uda --cls_dataset ImageFolder --n_classes 6 --mode test
```

### ReMixMatch

#### train model
```bash
python main.py --ssl_model remixmatch  --cls_dataset ImageFolder --n_classes 6 --mu 1
```
#### test
```bash
python main.py --ssl_model remixmatch --cls_dataset ImageFolder --n_classes 6 --mode test
```

### AdaMatch
将半监督学习和领域自适应相结合
#### train model
```bash
python main.py --ssl_model adamatch  --cls_dataset ImageFolder --n_classes 9 --mu 5
```
#### test
```bash
python main.py --ssl_model adamatch --cls_dataset ImageFolder --n_classes 9 --mode test
```

### 研究点一 SGRW 在9人数据集上进行实验

#### train model

```bash
python main.py --ssl_model sgrw  --cls_dataset ImageFolder --n_classes 9 --ema_mask_init_ep 0 --ema_mask_end_ep 50 --ema_mask_init_w 1.0 --ema_mask_end_w 3.0
```
#### test
```bash
python main.py --ssl_model sgrw --cls_dataset ImageFolder --n_classes 9 --mode test
```


### 研究点二 SGWP 在6人数据集上进行实验

#### train model

```bash
python main.py --ssl_model sgwp --cls_dataset ImageFolder --n_classes 6 
```
#### test
```bash
python main.py --ssl_model sgwp --cls_dataset ImageFolder --n_classes 6 --mode test
```
## 可视化

1.装grad-cam库时，指定版本号1.2.9  
2.attentionmap featuremap filter train最后出或者test出，且须添加测试图片于--image_path指定路径中  
3.需添加inference输出embedding及lbl，再调用save_result_image函数出图  
注：函数中参数参照open_set中任意网络；model_layer_list选项与各自网络结构相关，必须包含卷积层如果不知道自己网络的各层名称，可导入torchextractor库并使用print(tx.list_module_names(self.model))来打印各层的名称
注：若要使用注意力图、特征图、卷积核图、增强后图像，需要确定输入的一张图像，请在保存vis输出的output文件夹中手动建立一个test文件夹，并放入一张测试图像；
或者任选测试图路径请在使用lib/opt.py中确定输入图像文件路径。  
### 监视器monitor使用：  
注：要先打开visdom再跑程序  

1.在终端对应环境导入库 
```
pip install visdom  
```
2.在终端打开visdom服务 
```
 python -m visdom.server  
 ```
3.在浏览器导航栏输入弹出的网址（部署端口号默认）   
```
http://localhost:8097  
```
4.另打开一个终端，运行程序即可（命令行需设置--control_monitor为1，且--visdom_port为8097）  
### 可视化图像种类与外部传参对应

出图								--control_save_img_type
t-SNE图								t-SNE
parameter（loss）-epoch的图	      	 valueepoch
metric-epoch的图					 metricepoch
注意力图							 attentionmap
特征图								 featuremap
卷积核图				 			  filter
增强后图像							  processed

监控								 --control_monitor
是				1（运行前，另开一个终端输入visdom打开接口）
否				0
### 可视化命令行举例
```
python main.py --ssl_model sgwp --cls_dataset ImageFolder --n_classes 9 --mode test --control_save_img_type attentionmap featuremap filter valueepoch metricepoch 
```
