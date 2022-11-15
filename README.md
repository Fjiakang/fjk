# Semi-Supervised-Learning

# 目录
- [环境配置](#环境配置)
- [File directory description](#代码框架)
- [补充](#补充)
- [研究点一（sgrw）数据集说明](#数据集说明)
- [研究点二（sgpw）数据集说明](#数据集说明)
- [Data setting](#数据集说明)
- [Main parameters used in our algorithms](#算法链接)
- [命令行举例](#运行)
	- [算法](#不同算法)
    	- [训练](#训练)
    	- [测试](#测试)
	- [实际用例及公共参数注释](#实际用例及公共参数注释)
- [研究点一命令行](#运行)
- [研究点二命令行](#运行)


## 配置环境

```
* python >= 3.6
* cuda >= 11.0 
grad-cam = 1.2.9
```

## File directory description

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


## 补充：
```
训练周期包含epoch eval_step total_steps，其中step与iter意义相同，且一般是通过设定iter的数目来设定训练次数。
```

## 研究点一（sgrw）数据集说明
```
该数据集为9人伪装数据集（在我的论文中我将该数据集称为9人身份实测数据集）。该数据集在NAS上已经整理好，data分为train和test。
运行时，请将NAS上 “研究点一数据/data” 文件夹直接下载到算法工程文件夹中，其中data和libs、cls_dataset、Vis等同级别。
所有的微多普勒频谱图都是有标签数据，但是在训练过程中会忽略许多数据的标签，将他们当作无标签数据看待。
注：train中包含三个压缩文件，cj1，cj2，cj3代表不同的有标签/无标签数据的配比情况，将它们全部解压到train内，会得到3组有标/无标文件夹，共6个文件夹。test中只包含一个压缩包，开压缩即可。
每组文件夹的调用都在算法的libs/leida_image.py中对cj==1 cj==2 cj==3进行if-else if判断来选择(这样不太好，若能改进就好了)，opt.py中设定默认的cj为2，也就是主实验。cj1对应这论文中的case(i)。cj3对应论文中的case(ii)
终端中添加输入参数--cj 1就是利用cj 1进行实验。训练数据有三组，测试数据中只有一组。Eg.
FixMatch算法
python main.py --ssl_model fixmatch --cls_dataset ImageFolder --n_classes 9 --cj 1
python main.py --ssl_model fixmatch --cls_dataset ImageFolder --n_classes 9 --cj 2
python main.py --ssl_model fixmatch --cls_dataset ImageFolder --n_classes 9 --cj 3
注意：.sh文件中都未添加cj的设置，也就是默认都是2，若要尝试1或3的情形，则自行复制添加即可。
```
## 研究点二（sgwp）数据集说明
```
研究点二包含三个数据集：①随机行走数据集②全向身份识别数据集③全向动作检测数据集，每个数据集都是多分类，各有六个类。三个数据集在NAS上已经整理好，data分为train和test。
运行时，数据文件的放置同“研究点一数据集说明.docx”，也是想在哪个数据集上实验就下载哪个数据集并解压放置到对应的位置。
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

## Data setting
```
半监督学习的输入数据包括有标签数据与无标签数据，因此准备两个dataloader进行训练，并在其中设定不同的数据增强方法(随算法改变而改变)
各个算法model.py中都添加了对数据增强的设定，包含对有标签数据和无标签数据的预处理。其涉及的参数包括
"--labeled_aug":表示对有标签数据进行增强方式的选择，包括weak，strong，normalize，nothing
"--un_img1_weak"：表示对无标签图像1进行增强方式的选择，若为True则是weak，否则nothing
"--un_img2_strong"：表示对无标签图像2进行增强方式的选择，若为True则是strong，否则weak

在半监督学习中存在强增强与弱增强的设定，weak弱增强则是简单的镜像、翻转的结合，而强增强则在弱增强的基础上包含额外的增强方式，比如RandAugment、CTAugment、AutoAugment、Mixup等，此库中的strong一般指RandAugment，因为该方法比较简洁且效果更佳。
```
## Main parameters used in our algorithms
```
算法链接:[Mean Teacher](https://dl.acm.org/doi/pdf/10.5555/3294771.3294885) | [FixMatch](https://arxiv.org/ftp/arxiv/papers/2001/2001.07685.pdf) | [MixMatch](https://arxiv.org/abs/1905.02249) | [Pseudo-Labeling](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.664.3543&rep=rep1&type=pdf) | [ReMixMatch](https://arxiv.org/pdf/1911.09785.pdf) | [UDA](https://arxiv.org/pdf/1904.12848.pdf) | [VAT](https://ieeexploreieee.53yu.com/abstract/document/8417973) 
```
## 命令行举例
### Mean Teacher
Code for the paper: "[Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://dl.acm.org/doi/pdf/10.5555/3294771.3294885)" by 
Tarvainen A, Valpola H

#### train model
```bash
python main.py --ssl_model meanteacher --weight_decay 5e-4 --lr 0.0003 --cls_dataset ImageFolder --n_classes 6 --mu 1 --total_steps 30000 --eval_step 1000 
```
#### test
```bash
python main.py --ssl_model meanteacher --cls_dataset ImageFolder --n_classes 6 --mode test 
```

### MixMatch
This is an unofficial PyTorch implementation of [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)by
Berthelot D, Carlini N, Goodfellow I, et al.

#### train model
```bash
python main.py --ssl_model mixmatch  --cls_dataset ImageFolder --n_classes 9 --mu 1
```
#### test
```bash
python main.py --ssl_model mixmatch --cls_dataset ImageFolder --n_classes 9 --mode test
```
### Pseudo-Label

Code for the paper: "[Pseudo-Label: The simple and efficient semi-supervised learning method for deep neural networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.664.3543&rep=rep1&type=pdf)" by 
Lee D H

#### train model

```bash
python main.py --ssl_model pl --mu 1 --cls_dataset ImageFolder --n_classes 6
```
#### test
```bash
python main.py --ssl_model pl --cls_dataset ImageFolder --n_classes 6 --mode test
```
### VAT

Code for the paper: "[Virtual adversarial training: a regularization method for supervised and semi-supervised learning](https://ieeexploreieee.53yu.com/abstract/document/8417973)" by Miyato T, Maeda S, Koyama M, et al.

#### train model

```bash
python main.py --ssl_model vat --mu 1 --cls_dataset ImageFolder --n_classes 6
```
#### test
```bash
python main.py --ssl_model vat --cls_dataset ImageFolder --n_classes 6 --mode test
```
### specific parameter of VAT 
这些参数代表VAT中的loss权重以及VAT中特殊的perturbation size值
```python
"--entmin_weight",type=float, default=0.06,help='Entropy minimization weight'
"--vat_eps"，type=int, default=6, help='VAT perturbation size.'
```

### FixMatch

Code for the paper: "[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)" by 
Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel.

#### train model

```bash
python main.py --ssl_model fixmatch  --cls_dataset ImageFolder --n_classes 6
```
#### test
```bash
python main.py --ssl_model fixmatch --cls_dataset ImageFolder --n_classes 6 --mode test
```

## 研究点一 SGRW 在9人数据集上进行实验

#### train model

```bash
python main.py --ssl_model sgrw  --cls_dataset ImageFolder --n_classes 9 --ema_mask_init_ep 0 --ema_mask_end_ep 50 --ema_mask_init_w 1.0 --ema_mask_end_w 3.0
```
#### test
```bash
python main.py --ssl_model sgrw --cls_dataset ImageFolder --n_classes 9 --mode test
```
### specific parameter of ours 
这些参数代表新添加的之路中的ema_mask变化的epoch与变化的范围并且在变化后在剩余的epoch中一直保持ema_mask_end_w的值
```python
"--ema_mask_init_ep",type=int, default=0,help='Initial changing epoch of the ema mask'
"--ema_mask_end_ep",type=int, default=50,help='Final changing epoch of the ema mask'
"--ema_mask_init_w",type=float, default=1.0,help='Initial weight of the ema mask'
"--ema_mask_end_w",type=float, default=3.0,help='Final weight of the ema mask'
```

## 研究点二 SGWP 在6人数据集上进行实验

#### train model

```bash
python main.py --ssl_model sgwp --cls_dataset ImageFolder --n_classes 6 
```
#### test
```bash
python main.py --ssl_model sgwp --cls_dataset ImageFolder --n_classes 6 --mode test
```


### UDA
Pytorch Code for the paper: "[Unsupervised Data Augmentation](https://arxiv.org/pdf/1904.12848.pdf) by
by Xie Q, Dai Z, Hovy E, et al

#### train model
```bash
python main.py --ssl_model uda  --cls_dataset ImageFolder --n_classes 6  --mu 1 --lr 0.1
```
#### test
```bash
python main.py --ssl_model uda --cls_dataset ImageFolder --n_classes 6 --mode test
```


### ReMixMatch

Code for the paper: "[ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring](https://arxiv.org/abs/1911.09785)" by David Berthelot, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang, and Colin Raffel.

#### train model
```bash
python main.py --ssl_model remixmatch  --cls_dataset ImageFolder --n_classes 6 --mu 1
```
#### test
```bash
python main.py --ssl_model remixmatch --cls_dataset ImageFolder --n_classes 6 --mode test
```
### specific parameter of ReMixMatch 
相较于MixMatch，Remixmatch中额外添加了连个损失项，所以也加入了两个独有的损失的权重。
```python
"--lam_us",type=float, default=0.5,help='weight of Lus'
"--lam_rot",type=float, default=1.5,help='weight of Rotloss'
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

