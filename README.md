# 基于多模型协同过滤的机器学习方法

## 项目概述
本项目致力于开发一个餐厅评分预测算法，专注于通过多种机器学习技术对Yelp数据集上的餐厅进行评分预测。我们使用了包括奇异值分解SVD、余弦相似度、交替最小二乘法ALS、随机梯度下降SGD和随机森林等方法，以提供准确的评分预测并增强推荐系统的性能。

## 项目配置方法

1. **（推荐）安装Anaconda**

   - 访问Anaconda官方网站：[www.anaconda.com](https://www.anaconda.com)，从[下载页面](https://www.anaconda.com/download)获取安装程序。
   - 如果在**中国大陆**，由于网络问题，建议使用[清华大学Anaconda镜像站点](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)进行下载。
   - 完成安装后，按照[这个教程](https://blog.csdn.net/weixin_43914658/article/details/108785084)来配置环境变量，以确保可以从命令行运行Anaconda。

2. **进入项目目录**

   - 使用命令行界面，导航到您的项目文件夹。

3. **创建并激活虚拟环境**

   - 创建虚拟环境：在命令行中输入以下命令：

     ```bash
     conda create -n Yelp_env python=3.11
     ```

   - 激活虚拟环境：

     ```bash
     conda activate Yelp_env
     ```

4. **安装项目依赖**

   - 使用pip安装依赖。在项目目录中，有[`requirements.txt`](./requirements.txt)文件，列出了所有必需的Python库，使用以下命令安装依赖：

     ```bash
      pip install -r requirements.txt
     ```
     
   - 如果在**中国大陆**，考虑使用[清华大学PyPI镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)加速依赖安装：
   
  ```bash
     pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

## 贡献者

[孙逸青](mailto:william_syq@tju.edu.cn)

[张颢南](mailto:shu_1294491613@tju.edu.cn)

