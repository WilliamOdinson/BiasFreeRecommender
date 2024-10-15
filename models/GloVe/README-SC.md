# GloVe: 全局词向量表示

作者：[杰弗里·彭宁顿](http://stanford.edu/~jpennin)、[理查德·索彻](http://www.socher.org/)、[克里斯托弗·D·曼宁](https://nlp.stanford.edu/~manning)

---

## 引言

GloVe 是一种用于获取词的向量表示的无监督学习算法。训练是在一个语料库的全局词-词共现统计数据上进行的，生成的表示揭示了词向量空间的有趣线性子结构。

## 入门（代码下载）

- 下载最新的[代码](https://github.com/stanfordnlp/GloVe)（根据[Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)授权）。
  查找“克隆或下载”
- 解压文件：解压 master.zip
- 编译源代码：cd GloVe-master && make
- 运行演示脚本：./demo.sh
- 有关进一步使用细节，请查阅随附的 README 文件，或提出[问题](https://nlb.stanford.edu/projects/glove/#discuss)

## 下载预训练词向量

- 预训练词向量。这些数据在[公共领域奉献许可证](http://opendatacommons.org/licenses/pddl/)v1.0 下提供，许可证全文可在此查看：http://opendatacommons.org/licenses/pddl/1.0/

  - [维基百科 2014](http://dumps.wikimedia.org/enwiki/20140102/) + [Gigaword 5](https://catalog.ldc.upenn.edu/LDC2011T07) (60亿标记，40万词汇量，不区分大小写，50d、100d、200d和300d向量，下载822 MB)：[glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)
  - 公共语料库 (420亿标记，190万词汇量，不区分大小写，300d向量，下载1.75 GB)：[glove.42B.300d.zip](https://nlp.stanford.edu/data/glove.42B.300d.zip)
  - 公共语料库 (8400亿标记，220万词汇量，区分大小写，300d向量，下载2.03 GB)：[glove.840B.300d.zip](https://nlp.stanford.edu/data/glove.840B.300d.zip)
  - Twitter (20亿推文，270亿标记，120万词汇量，不区分大小写，25d、50d、100d和200d向量，下载1.42 GB)：[glove.twitter.27B.zip](https://nlp.stanford.edu/data/glove.twitter.27B.zip)

- Ruby [脚本](https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb) 用于预处理 Twitter 数据

- ```ruby
  # Ruby 2.0
  # Reads stdin: ruby -n preprocess-twitter.rb
  #
  # Script for preprocessing tweets by Romain Paulus
  # with small modifications by Jeffrey Pennington
  
  def tokenize input
  
  	# Different regex parts for smiley faces
  	eyes = "[8:=;]"
  	nose = "['`\-]?"
  
  	input = input
  		.gsub(/https?:\/\/\S+\b|www\.(\w+\.)+\S*/,"<URL>")
  		.gsub("/"," / ") # Force splitting words appended with slashes (once we tokenized the URLs, of course)
  		.gsub(/@\w+/, "<USER>")
  		.gsub(/#{eyes}#{nose}[)d]+|[)d]+#{nose}#{eyes}/i, "<SMILE>")
  		.gsub(/#{eyes}#{nose}p+/i, "<LOLFACE>")
  		.gsub(/#{eyes}#{nose}\(+|\)+#{nose}#{eyes}/, "<SADFACE>")
  		.gsub(/#{eyes}#{nose}[\/|l*]/, "<NEUTRALFACE>")
  		.gsub(/<3/,"<HEART>")
  		.gsub(/[-+]?[.\d]*[\d]+[:,.\d]*/, "<NUMBER>")
  		.gsub(/#\S+/){ |hashtag| # Split hashtags on uppercase letters
  			# TODO: also split hashtags with lowercase letters (requires more work to detect splits...)
  
  			hashtag_body = hashtag[1..-1]
  			if hashtag_body.upcase == hashtag_body
  				result = "<HASHTAG> #{hashtag_body} <ALLCAPS>"
  			else
  				result = (["<HASHTAG>"] + hashtag_body.split(/(?=[A-Z])/)).join(" ")
  			end
  			result
  		} 
  		.gsub(/([!?.]){2,}/){ # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
  			"#{$~[1]} <REPEAT>"
  		}
  		.gsub(/\b(\S*?)(.)\2{2,}\b/){ # Mark elongated words (eg. "wayyyy" => "way <ELONG>")
  			# TODO: determine if the end letter should be repeated once or twice (use lexicon/dict)
  			$~[1] + $~[2] + " <ELONG>"
  		}
  		.gsub(/([^a-z0-9()<>'`\-]){2,}/){ |word|
  			"#{word.downcase} <ALLCAPS>"
  		}
  
  	return input
  end
  
  puts tokenize($_)
  ```

## 引用 GloVe

杰弗里·彭宁顿，理查德·索彻，克里斯托弗·D·曼宁。2014年。[GloVe: 全局词向量表示](https://nlp.stanford.edu/pubs/glove.pdf)。[[pdf](https://nlp.stanford.edu/pubs/glove.pdf)] [[bib](https://nlp.stanford.edu/pubs/glove.bib)]

```latex
@inproceedings{pennington2014glove,
  author = {Jeffrey Pennington and Richard Socher and Christopher D. Manning},
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  title = {GloVe: Global Vectors for Word Representation},
  year = {2014},
  pages = {1532--1543},
  url = {http://www.aclweb.org/anthology/D14-1162},
}	
```

## 亮点

**1. 最近邻**

两个词向量之间的欧几里得距离（或余弦相似度）提供了一种有效的方法来衡量相应词汇的语言或语义相似性。有时，根据这个度量找到的最近邻词揭示了一些稀有但相关的词汇，这些词汇超出了普通人的词汇范围。例如，以下是与目

标词 *frog*（青蛙）最接近的词：

1. *frog*（青蛙）
2. frogs（青蛙们）
3. toad（蟾蜍）
4. litoria（一种青蛙属）
5. leptodactylidae（狭趾蛙科）
6. rana（蛙属）
7. lizard（蜥蜴）
8. eleutherodactylus（赤蛙属）

**2. 线性子结构**

用于最近邻评估的相似度度量产生一个量化两个词相关性的单一标量。这种简单性可能是有问题的，因为两个给定的词几乎总是表现出比单一数字能够捕捉的更复杂的关系。例如，*man*（男人）和 *woman*（女人）可以被认为是相似的，因为这两个词都描述了人类；另一方面，这两个词通常被认为是对立的，因为它们强调人类不同的主要轴向。

为了在量化方式中捕捉区分 *man* 和 *woman* 所必需的细微差别，模型需要关联不止一个数字到词对。一种自然而简单的为词对分配一组辨别数字的方法是两个词向量之间的向量差。GloVe的设计目标是使这些向量差尽可能多地捕捉两个词并列使用时指定的含义。

**训练**

GloVe 模型是在一个全局词-词共现矩阵的非零条目上训练的，该矩阵统计了语料库中词汇的共现频率。填充这个矩阵需要遍历整个语料库一次以收集统计数据。对于大型语料库，这一过程可能在计算上非常昂贵，但这是一次性的前期成本。后续的训练迭代要快得多，因为非零矩阵条目的数量通常远小于语料库中的词汇总数。

这个包提供的工具自动化了输入模型的共现统计数据的收集和准备工作。核心训练代码与这些预处理步骤分离，可以独立执行。

**模型概览**

GloVe 本质上是一个带有加权最小二乘目标的对数双线性模型。模型的主要直觉是简单的观察：词-词共现概率的比率有潜力编码某种形式的含义。例如，考虑目标词 *ice*（冰）和 *steam*（蒸汽）与词汇表中各种探针词的共现概率。这里是一个来自60亿词语料库的实际概率：

| 概率与比率             | $k = solid$（固体） | $k = gas$（气体）    | $k = water$（水）    | $k = fashion$（时尚） |
| ---------------------- | ------------------- | -------------------- | -------------------- | --------------------- |
| $P(k \mid 冰)$ |               $1.9 \times 10^{-4}$ | $6.6 \times 10^{-5}$ | $3.0 \times 10^{-3}$  |$1.7 \times 10^{-5}$|
| $P(k \mid 蒸汽)$ |               $2.2 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | $2.2 \times 10^{-3}$  |$1.8 \times 10^{-5}$|
| $P(k \mid 冰)/P(k \mid 蒸汽)$ |  $8.9$                | $8.5 \times 10^{-2}$ | $1.36$               |$0.96$|

如预期，*ice*（冰）与 *solid*（固体）的共现频率比与 *gas*（气体）的更频繁，而 *steam*（蒸汽）与 *gas* 的共现频率比与 *solid* 更频繁。这两个词与它们共有的属性 *water*（水）频繁共现，与无关的词 *fashion*（时尚）不频繁共现。只有在概率比中，非歧视性词如 *water* 和 *fashion* 的噪音才被消除，从而使得大的值（远大于1）与冰特有的属性相关，小的值（远小于1）与蒸汽特有的属性相关。这样，概率比就编码了与热力学阶段的抽象概念相关的某种粗略形式的含义。

训练目标是学习词向量，使得它们的点积等于词共现概率的对数。由于对数的比率等于对数的差，这个目标将共现概率的比率（对数）与词向量空间中的向量差联系起来。由于这些比率可以编码某种形式的含义，这种信息也被编码为向量差。因此，生成的词向量在词类比任务上表现非常好，如[word2vec](http://code.google.com/p/word2vec/)包中检查的任务。

**可视化**

GloVe 生成的词向量具有明显的带状结构，这在可视化时很明显：

![word_vectors](./assets/word_vectors.jpg)

水平带由于模型中的乘法交互是按组件进行的而产生。虽然有点积产生的加法交互，但通常很少有空间让单个维度交叉污染。

随着词频的增加，水平带变得更加明显。确实，随着词频的增加，可以看到一些长期趋势，这些趋势不太可能有语言来源。这一特征并非GloVe所独有——实际上，我不知道有哪个词向量学习模型可以避免这个问题。

垂直带，如围绕词230k-233k的带，是由于相关词（通常是数字）的局部密度相似而产生的。

**版本历史**

- [GloVe v.1.2](https://nlp.stanford.edu/software/GloVe-1.2.zip)：代码中的小错误修复（内存，越界，错误）。现在评估代码也可在Python和Octave中使用。最大数据文件的UTF-8编码已修复。由Russell Stewart和Christopher Manning准备。2015年10月。
- [GloVe v.1.0](https://nlp.stanford.edu/software/GloVe-1.0.tar.gz)：原始发布。由Jeffrey Pennington准备。2014年8月。

**错误/问题/讨论**

**GitHub**：GloVe 在 [GitHub](https://github.com/stanfordnlp/GloVe) 上。对于错误报告和补丁，最好使用GitHub的问题和拉取请求功能。

**Google Group**：Google Group [globalvectors](https://groups.google.com/forum/#!forum/globalvectors) 可用于提问和关于GloVe的一般讨论。