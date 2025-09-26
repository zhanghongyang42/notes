# 关键词提取

关键词提取技术，包括有监督（部分打标签，训练，人工把预测正确关键词的数据再加入训练集，继续训练），无监督（TF-IDF、TextRank、基于主题）



### bag of words

词袋模型：把每篇文档出现的单词按照顺序排列，统计出每个token（一般就是一个单词）的出现次数，组成向量。

词袋模型忽略了词序语法近义等，每个词相互独立存在。

```python
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())
print(X.toarray())

#ngram_range默认1个词为1个token，可以设置1，2，3个词都是1个token，这样可以把词的顺序信息训练进来
bigram_vectorizer = CountVectorizer(ngram_range=(1, 3))
```



### TF-IDF

关键词提取技术的一种，通过计算每个词在文章中出现的频率*该词的重要性来确定关键词。



TF-IDF = TF*IDF

![20180806135836378](https://raw.githubusercontent.com/zhanghongyang42/images/main/20180806135836378.png)

![20180806140023860](https://raw.githubusercontent.com/zhanghongyang42/images/main/20180806140023860.png)



明显看出，词袋模型的结果可以直接用来计算TF-IDF，所以实现为CountVectorizer+TfidfTransformer   或者  TfidfVectorizer。



##### TfidfTransformer

```python
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)

#词袋模型结果，一个列表代表一篇文章，共有6篇文章，每一列代表一个词，3代表第一个词在第一篇文章里出现了3次。
counts = [[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]]

#每一篇文章的每一个词都可以计算出tf-idf，最后把结果归一化
tfidf = transformer.fit_transform(counts)
print(tfidf.toarray())
```



##### TfidfVectorizer

CountVectorizer  +  TfidfTransformer

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(corpus)
```



### TextRank

##### PageRank

TextRank 的思想来源于 PageRank。

PageRank：https://www.cnblogs.com/jpcflyer/p/11180263.html

```python
#PageRank 过程
给每个点（网页）初始权重。
写出转移矩阵（转移矩阵的理论基础就是那个pr计算公式，按改良公式进行转移得保证总量不变）。
根据转移矩阵更新初始权重。直至权重不再变化。
不再变化的权重就是那个网页的pr值（pr值不再变化的时候就完全反应了累积过后的【转移关系(用户意愿)】）。
```



##### TextRank

https://www.cnblogs.com/motohq/p/11887420.html

TextRank 与 PageRank相比，不同是每个词语之间都是双向边，且边有权重（双向边代表在同一句子中出现，边的权重代表出现次数或者相似度）。表现为 转移矩阵或者说理论公式不同。

 

本质是用词之间的关系找出关键词。所以 这种计算方法一定要过滤掉停用词。

```python
#TextRank 过程
1.把文本 按 句子进行分割
2.对每个句子进行 分词和词性标注处理，并过滤掉停用词，只保留指定词性的单词。
3.按PageRank 过程 构建图，计算出每个词的权重。
4.选topN 作为 候选关键词。
5.看候选关键词是否在文本中可以组成关键词组，把关键词组也作为关键词。
```

```python
import pandas as pd
import numpy as np
from textrank4zh import TextRank4Keyword, TextRank4Sentence

# 关键词抽取
def keywords_extraction(text):
    tr4w = TextRank4Keyword(allow_speech_tags=['n', 'nr', 'nrfg', 'ns', 'nt', 'nz'])
    # allow_speech_tags   --词性列表，用于过滤某些词性的词
    tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters', edge_source='no_stop_words',
                 pagerank_config={'alpha': 0.85, })
    # text    --  文本内容，字符串
    # window  --  窗口大小，int，用来构造单词之间的边。默认值为2
    # lower   --  是否将英文文本转换为小写，默认值为False
    # vertex_source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点
    #                -- 默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'
    # edge_source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点之间的边
    #              -- 默认值为`'no_stop_words'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。边的构造要结合`window`参数

    # pagerank_config  -- pagerank算法参数配置，阻尼系数为0.85
    keywords = tr4w.get_keywords(num=6, word_min_len=2)
    # num           --  返回关键词数量
    # word_min_len  --  词的最小长度，默认值为1
    return keywords

# 关键短语抽取
def keyphrases_extraction(text):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters', edge_source='no_stop_words',
                 pagerank_config={'alpha': 0.85, })
    keyphrases = tr4w.get_keyphrases(keywords_num=6, min_occur_num=1)
    # keywords_num    --  抽取的关键词数量
    # min_occur_num   --  关键短语在文中的最少出现次数
    return keyphrases

# 关键句抽取
def keysentences_extraction(text):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text, lower=True, source='all_filters')
    # text    -- 文本内容，字符串
    # lower   -- 是否将英文文本转换为小写，默认值为False
    # source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来生成句子之间的相似度。
    # 		  -- 默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'
    # sim_func -- 指定计算句子相似度的函数

    # 获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要
    keysentences = tr4s.get_key_sentences(num=3, sentence_min_len=6)
    return keysentences

if __name__ == "__main__":
    text = "来源：中国科学报本报讯（记者肖洁）又有一位中国科学家喜获小行星命名殊荣！4月19日下午，中国科学院国家天文台在京举行“周又元星”颁授仪式，" \
           "我国天文学家、中国科学院院士周又元的弟子与后辈在欢声笑语中济济一堂。国家天文台党委书记、" \
           "副台长赵刚在致辞一开始更是送上白居易的诗句：“令公桃李满天下，何须堂前更种花。”" \
           "据介绍，这颗小行星由国家天文台施密特CCD小行星项目组于1997年9月26日发现于兴隆观测站，" \
           "获得国际永久编号第120730号。2018年9月25日，经国家天文台申报，" \
           "国际天文学联合会小天体联合会小天体命名委员会批准，国际天文学联合会《小行星通报》通知国际社会，" \
           "正式将该小行星命名为“周又元星”。"
   
	# 关键词抽取
    keywords =keywords_extraction(text)
    print(keywords)

    # 关键短语抽取
    keyphrases =keyphrases_extraction(text)
    print(keyphrases)

    # 关键句抽取
    keysentences =keysentences_extraction(text)
    print(keysentences)
```



### 主题模型

主题模型是生成模型。生成模型与判别模型的区别在于：判别模型直接计算出概率，生成模型比较各个可能的概率，选最大的。

主题模型包括 LSA、pLSA、LDA、HDP、LDA2Vec。



##### LSA

https://zhuanlan.zhihu.com/p/46376672

```python
#LSA过程
1.首先计算出 TF-IDF
2.进行svd矩阵分解 特征向量
3.利用余弦相似性找出相似文本
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#读数据
from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
dataset.target_names

#删除符号与短词
news_df = pd.DataFrame({'document':documents})
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

#删除停用词
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
detokenized_doc = []
for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
news_df['clean_doc'] = detokenized_doc

#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english',max_df = 0.5, max_features= 1000,smooth_idf=True)
X = vectorizer.fit_transform(news_df['clean_doc'])

# SVD
from sklearn.decomposition import TruncatedSVD
svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)

#主题排序
terms = vectorizer.get_feature_names()
for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    print("Topic "+str(i)+": ",end='')
    for t in sorted_terms:
        print(t[0],end=' ')
    print(" ")
```

