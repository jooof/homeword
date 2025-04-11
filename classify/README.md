# 垃圾邮件分类器 

一个基于朴素贝叶斯算法的中文邮件分类器，可自动识别垃圾邮件与普通邮件。

## 主要功能
- 文本预处理：自动过滤无效字符与短词
- 词频特征提取：自动构建高频词特征库（Top 100）
- 机器学习模型：多项式朴素贝叶斯分类器
- 邮件分类预测：支持对新邮件进行自动分类

## 一、库引入分析
import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
re：用于正则表达式过滤无效字符

jieba：中文分词工具

Counter：统计词频

numpy：处理数值向量

MultinomialNB：适用于离散特征（如词频）的朴素贝叶斯分类器

## 二、核心函数解析
### 1. 文本处理函数 get_words()
def get_words(filename):
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)  # 过滤标点数字
            line = cut(line)  # 中文分词
            line = filter(lambda word: len(word) > 1, line)  # 过滤单字
            words.extend(line)
    return words
输入：文本文件路径

输出：过滤后的词语列表
处理流程：
移除空白字符
正则表达式过滤指定符号和数字
使用结巴分词
过滤长度≤1的词语

### 2. 高频词提取函数 get_top_words()
def get_top_words(top_num):
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    # ...（遍历文件构建词库）
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]
功能：统计训练集中出现频率最高的词语

实现细节：
使用chain(*all_words)将多层列表展平
Counter.most_common()获取前N个高频词
输出：前100个高频词列表


### 构建词向量
vector = []
for words in all_words:
    word_map = list(map(lambda word: words.count(word), top_words))
    vector.append(word_map)
vector = np.array(vector)
逻辑说明：
为每个邮件创建特征向量 向量元素表示对应高频词在邮件中的出现次数

[0, 2, 0, ..., 1]  # 每个数字对应top_words中的词频
### 四、模型训练部分

labels = np.array([1]*127 + [0]*24)  # 前127为垃圾邮件
model = MultinomialNB()
model.fit(vector, labels)
标签分配：

0-126.txt标记为1（垃圾邮件）

127-150.txt标记为0（普通邮件）

### 算法选择：

多项式朴素贝叶斯适合处理离散型特征（词频统计）

## 五、预测函数分析
def predict(filename):
    words = get_words(filename)
    current_vector = np.array(tuple(map(lambda word: words.count(word), top_words)))
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'
实现流程：
对新邮件进行相同预处理
生成与训练集相同维度的词频向量
使用训练好的模型进行预测

### 六、潜在优化建议
停用词处理：高频词可能包含无意义词汇（"的"，"是"）

TF-IDF加权：替代简单词频统计

样本均衡：普通邮件样本较少（24 vs 127）

交叉验证：提高模型评估可靠性

特征选择：使用卡方检验等选择区分性词语

### 七、代码执行示例
print(predict('邮件_files/151.txt'))  # 输出：垃圾邮件/普通邮件
测试说明：
151-155.txt为独立测试文件未参与训练过程

### 分类作业
<img src="https://github.com/jooof/homeword/blob/master/img_2.png?raw=true" width="800" alt="截图三">