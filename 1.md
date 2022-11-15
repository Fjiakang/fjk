# Classification

面向单分类、多分类以及开集识别的通用分类框架

<!-- PROJECT SHIELDS -->
<!-- 
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->

## 目录

- [上手指南](#上手指南)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装步骤](#安装步骤)
- [文件目录说明](#文件目录说明)
- [开发遵循规范](#开发遵循规范)
- [使用规范](#使用规范)
- [开发的架构](#开发的架构)
- [部署](#部署)
- [使用到的框架](#使用到的框架)
- [贡献者](#贡献者)
- [如何参与开源项目](#如何参与开源项目)
- [版本控制](#版本控制)
- [作者](#作者)
<!-- - [鸣谢](#鸣谢) -->

### 上手指南

###### 开发前的配置要求

1. python >= 3.6
2. cuda >= 11.0

###### **安装步骤**

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo

```sh
git clone https://github.com/lab542-tju/Classification.git
```

### 文件目录说明
eg:

```
┌── Data
│  ├── /mnist/
│  ├── /radar/
│  │  ├── /MCL/
│  │  │  ├── /train/
│  │  │  │  │  ├── /0/
│  │  │  │  │  ├── 1.jpg
│  │  │  │  │  ├── 2.jpg
│  │  │  │  │  └── ...
│  │  │  │  ├── /1/
│  │  │  │  └── ...
│  │  │  └── /test/
│  │  ├── /PID/
│  │  └── /.../
│  └── ...
├── Classification (本级目录)
│  ├── README.md
│  ├── /libs/
│  │  ├── base_model_d.py
│  │  ├── base_model_t.py
│  │  ├── data.py
│  │  ├── metric.py
│  │  ├── opt.py
│  │  ├── Visualizer.py
│  │  ├── loss.py(待删除)
│  │  ├── utils.py（待删除）
│  ├── /one_class/
│  │  ├── ALOCC.py
│  │  ├── ...
│  ├── /multi_class/
│  │  ├── kimnet.py
│  │  ├── ...
│  ├── /open_set/
│  │  ├── OpenGAN.py
│  │  ├── ...
│  └── main.py
└── Output
```
### 开发遵循规范 

1. 所有算法均以独立文件夹保存于不同分类文件夹（one_class，multi_class，open_set）的子目录下。

2. 当前算法文件夹的命名与算法类的命名一致。

3. 算法文件夹中网络结构的命名格式为：算法_network。

4. 算法文件夹中须包含复现文章的pdf文件，命名格式为：年份+出版物+题目。

5. 算法文件夹中模型文件前两行需注明算法来源文章，及标准引用格式。

### 使用规范

```sh
python main.py <one_class/multi_class/open_set> <--args1 args1_value> <--args2 args2_value>
```
主函数常用参数

```python
"cls_type", type=str,default="one_class",choices=['one_class', 'multi_class', 'open_set']
"--phase", type=str, default="train",choices=['train','test']
"--cls_dataroot", type=str, default="../data"
"--cls_dataset", type=str, default="out30"
"--cls_network", type=str, default="hmm"
"--cls_imageSize", type=int
"--cls_imageSize2", type=int, default=-1
"--nc", type=int, default=3
"--cls_batchsize", type=int, default=64
"--metric", nargs="+", default=['AUC']
"--metric_pos_label", type=str, default = '9'
"--metric_average", type=str
"--outf", type=str, default='../output'
"--control_save_end", type=int, default=0, help='save the weights on terminal(default False)'
"--control_print", action='store_true', help='print the results on terminal(default False)'
"--control_save", action='store_true', help='save the results to files(default False)'
"--load_weights", action='store_true', help='load parameters(default False)'
"--gray_image", action='store_true', help='convert image to grayscale(default False)'
"--aug_methods", nargs='+', default=[]
```

### 开发的架构 

暂无

### 部署

暂无

### 使用到的框架

- [Anaconda](https://www.anaconda.com/products/individual)
- [Pytorch](https://pytorch.org)

### 贡献者

暂无

### 如何参与项目

1. 项目克隆
   通过加速插件使用ssh克隆到本地电脑以形成本地仓库，以classification仓库为例：
```   
   git clone git@git.zhlh6.cn:lab542-tju/Classification.git '指定文件夹名（可省略）'
```   
2. 创建feature branch：
   每个远程仓库具有main、develop两个分支，其中main供使用，develop供修改。在每次调试时，需要在本地仓库上新建feature分支以减少在远程develop分支上的commit数量：
```
   git checkout -b feature
```
3. 修改代码：
   在修改代码后，需要先将修改提交至本地仓库的缓冲区：
```
   git add .
```
   之后将缓冲区的修改形成commit：
```
   git commit -m '版本名称'
```
4. 合并分支并删除：
   自行测试feature分支修改无误后，切换到develop分支：
```
   git checkout develop
```
   使用拉取命令以使本地仓库的develop分支与远程仓库同步:
```
   git pull
```
   合并本地feature分支
```
   git merge feature
```
   将合并后的develop分支进行上传
```
   git push origin develop
```
   删除本地feature分支
```
   git branch -d feature
```
*5. 回退版本：*

当修改出现问题时，想要回退版本，首先调用：
```
   git log
```
   找到git log中想要回退到的版本号，使用以下命令回退：
```
   git reset --hard '版本号'
```
*6. .gitignore：*

   当文件夹中出现代码不需要的文件时，避免上传，在根目录下.gitignore中添加该文件路径或格式，写法如：
```
   ./libs/__pycache__
   *.pyc
```
### 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。

### 作者

xxx@xxxx

知乎:xxxx  &ensp; qq:xxxxxx    

 *您也可以在贡献者名单中参看所有参与该项目的开发者。*

<!-- ### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE.txt](https://github.com/lab542-tju/Classification/blob/master/LICENSE.txt) -->

<!-- ### 鸣谢


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)
- [Animate.css](https://daneden.github.io/animate.css)
- [xxxxxxxxxxxxxx](https://connoratherton.com/loaders) -->

<!-- links -->
<!-- [your-project-path]:lab542-tju/Classification
[contributors-shield]: https://img.shields.io/github/contributors/lab542-tju/Classification.svg?style=flat-square
[contributors-url]: https://github.com/lab542-tju/Classification/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/lab542-tju/Classification.svg?style=flat-square
[forks-url]: https://github.com/lab542-tju/Classification/network/members
[stars-shield]: https://img.shields.io/github/stars/lab542-tju/Classification.svg?style=flat-square
[stars-url]: https://github.com/lab542-tju/Classification/stargazers
[issues-shield]: https://img.shields.io/github/issues/lab542-tju/Classification.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/lab542-tju/Classification.svg
[license-shield]: https://img.shields.io/github/license/lab542-tju/Classification.svg?style=flat-square
[license-url]: https://github.com/lab542-tju/Classification/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian -->
