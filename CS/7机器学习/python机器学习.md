scikit-learn：https://scikit-learn.org/stable/

吴恩达机器学习视频：https://www.bilibili.com/video/BV1Bq421A74G（差3.1以后 异常检测，推荐算法，强化学习）

吴恩达机器学习笔记：http://www.ai-start.com/ml2014



# 概述

### 建模步骤

读取数据

探索性数据分析

特征工程

数据划分

模型训练

模型评价

保存部署



交叉验证就是多次划分数据集进行模型训练，模型评价，可以得到一个更没有偶然性的评价。

超参数搜索使用不同组的参数，每组参数进行交叉验证。更稳定的选出模型最好的参数，也同时训练出了最好的模型，也得到了模型评价。

保存部署模型，可以直接保存模型。也可以把特征工程和训练好的模型打成pipeline，再保存部署。



### 数据介绍

一般处理使用的数据都是 结构化数据。

结构化数据分为**数值型**和**类别型**，数值型又有**连续型**和**离散型**。



### tips

1、如果数据量较大，程序运行时间长，可以在过程中持久化中间数据和模型，避免从头运行。

2、大数据量的特征工程比较容易，小数据量的特征工程的很多操作不具有泛化性。

3、取得效果最重要的是 定义的问题的合理性 和 数据，保证这两点。



# 读数据

### 表格

读数据

```python
df = pd.read_df = foo.csv',sep=',',encoding='UTF-8')
df = pd.read_excel('foo.xlsx')
```

写数据

```python
df.to_csv('foo.csv')
```



### Oracle

读oracle

```python
import cx_Oracle

db = cx_Oracle.connect( 'ods/ods@198.1.6.67:1521/orcl',encoding='UTF-8',nencoding='UTF-8' )
cursor = db.cursor()

sql = "select column_name from user_tab_columns" 
cursor.execute(sql)

read = list(cursor.fetchall())

ls=[]
for i in read:
    ls.append(list(i)[0])
data= pd.DataFrame(read,columns=ls)

cursor.close()
db.close()
```



### Hive

读取hive

```python
from pyhive import hive

time_start = time.time()

# 获取数据
conn = hive.Connection(host="10.0.1.192", port="10000",database='app')
cursor = conn.cursor()
query_sql = 'select * from app_user_search_train'
cursor.execute(query_sql)
result = cursor.fetchall()
cursor.close()
conn.close()

# 获取列名称
conn = hive.Connection(host="10.0.1.192", port="10000",database='app')
cursor = conn.cursor()
query_sql = 'desc app_user_search_train'
cursor.execute(query_sql)
col_names = cursor.fetchall()
cursor.close()
conn.close()

# 组合成dataframe
col_names_list = []
for tup in col_names:
    col_names_list.append(tup[0])
col_names_list = col_names_list[0:-5]

df = pd.DataFrame(result,columns=col_names_list)

print(time.time()-time_start)

del result
```



# EDA

探索性数据分析 的过程和 数据预处理、特征工程是交替进行的，EDA的主要目的就是理解数据和业务。



### AUTO_EDA

查看缺失值，结合分析出异常值，0值，查看高基数特征，查看整体相关性

```python
import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

df = pd.read_csv('churn.csv')

# pycharm执行代码
pfr = pandas_profiling.ProfileReport(df)
pfr.to_file("./example.html")
 
# 在jupyter执行代码
# pandas_profiling.ProfileReport(df)
```



### 基本查看

```python
df.head()
df.tail()
df.sample()

df.shape

df.info()

df.dtypes

df.describe()
```



### 正负样本比

```python
df_vc = df['Survived'].value_counts()
```



### 单列数据分析

频度统计

```python
df['a'].value_counts().sort_values()
df['a'].value_counts().count()
```

数据分布

```python
sns.kdeplot(df[col], label=str(col))
```



### 相关系数原理

相关系数主要描述变量之间的线性关系，而相似度（距离）则关注具体数据对象之间的相似程度。



#### 相关系数

对协方差进行标准化后的指标，取值范围为 $[-1, 1]$。计算公式为：
$$
\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}
$$
其中 $\sigma_X$ 和 $\sigma_Y$ 分别为 X和 Y的标准差。



标准差与原始数据具有相同的量纲，所以除以标准差，就消除了量纲。

实际上，将xy标准化后，求协方差，即是相关系数。



#### 皮尔逊相关系数

皮尔逊相关系数就是相关系数。
$$
r = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2 \sum_{i=1}^n (Y_i - \bar{Y})^2}}
$$


当皮尔逊相关系数接近1时,两个数据成正相关;当皮尔逊相关系数接近-1时,两个数据成负相关;当皮尔逊相关系数接近0时,两个数据没有明显关系.



#### 斯皮尔曼相关系数

斯皮尔曼等级相关系数（Spearman's Rank Correlation Coefficient）用于衡量两个变量的秩（排序）之间的单调关系。


$$
\rho = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n(n^2 - 1)}
$$

- $d_i$是第 i 对变量的秩差，即两个变量在排序时的排名差。比如 x1 在x中 排第3， y1 在y中 排第1，则d1就是2。
- n是样本的数量。



公式解释：

- **分子部分**：计算所有秩差的平方和，反映了变量之间偏离完美单调关系的程度。
- **分母部分**：进行标准化，调整样本数量对相关系数的影响。
- **整体意义**：斯皮尔曼相关系数的值范围是 -1 到 1，值越接近 1 或 -1，表示两个变量之间的单调关系越强。值为 0 时表示没有单调关系。



### 特征间相关性分析

0-0.09为没有相关性,0.5-1.0为强相关

```python
# 所有相关性，类别型数据不计算
df.corr()

# 相关性大于0.3的特征
temp = df.corr()
high_corr = []
for i in list(itertools.combinations(temp.columns,2)):
    if abs(temp.loc[i[0],i[1]])>0.3:
        high_corr.append(i)
print(high_corr)
del temp

# 整体热力图
import seaborn as sns
import matplotlib.pyplot as plt
def heatmap_plot(df):
    dfcorr = df.corr()
    plt.figure()
    sns.heatmap(dfcorr, annot=True, vmax=1, square=True, cmap="Blues")
    plt.show()
heatmap_plot(df)

# 相关性大于正负0.3的热力图
temp = df.corr()
temp = temp[(abs(temp)>0.3) & (abs(temp)<1)]
for i in temp:
    if len(temp[i].value_counts())==0:
        temp.drop([i],axis=1,inplace=True)
        temp.drop([i],axis=0,inplace=True)
temp = list(temp.columns)
temp = df[temp].corr()
sns.heatmap(temp)
del temp
```

```python
# 通过画核密度图区分
import seaborn as sns
import matplotlib.pyplot as plt

def kde_target(df,lable,var_name):
    plt.figure()
    for i in df[lable].unique():
        sns.kdeplot(df.loc[df[lable]==i, var_name], label=str(i))
    plt.xlabel(var_name);
    plt.ylabel('Density'); 
    plt.title('%s Distribution' % var_name)
    plt.legend()
    plt.show()


if (df[col].dtypes!='object') & (col!='Survived'):
	kde_target(df,'Survived',col)
```



# 特征工程1

### 类别不平衡

针对分类预测会出现类别不平衡的问题。



大部分模型的预测结果，需要一个阈值进行分类，才能得到分类结果。比如二分类的预测结果为【0,1】，当某个样本的输出大于0.5会被划分为正例，反之则反。

类别不平衡时，比如90%的label为0，那么无论特征怎样，全部预测为0，也会得到很好的准确性。导致模型不具有泛化性。



解决方法: 

1.可以选择调整阈值，使得模型对于较少的类别更为敏感 

2.选择合适的评估标准，比如ROC或者F1，而不是准确度(accuracy) 

3.通过采样（sampling）来调整数据的不平衡，可以和阈值调整同时使用

4.使用集成学习学习更复杂的关系，可以更好学习到少量label的特征



### 数据采样

采样会对数据带来偏差：过采样可能对少数数据进行过拟合（可搭配正则化），欠采样会丢失数据信息（可从业务角度采样）。

数据量足够的时候进行欠采样，不过一般过采样效果较好。



##### imbalanced-learn

![v2-c2446593bf7c85c73f3715904ee24957_b](https://raw.githubusercontent.com/zhanghongyang42/images/main/v2-c2446593bf7c85c73f3715904ee24957_b-1606381549945.jpg)



##### k-means 采样

```
基于k-means聚类过采样方法一般分为两步：

    首先分别对正负例进行K-means聚类
    聚类之后，对其中较小的簇进行上面的过采样方法扩充样本数量
    然后在进行正负类样本均衡扩充

该算法不仅可以解决类间不平衡问题，而且还能解决类内部不平衡问题。
```



##### 业务采样

根据需求，这里是根据时间分层采样，对每个月的数据进行欠采样。

```python
data_new = pd.DataFrame(columns=data.columns)
time = sorted(list(data.data_dt.unique()))
for i in range(len(time)):
    data_time = data[data.data_dt==time[i]]
    data_time_sample = data_time[data_time.y==0].sample(100000,axis=0)
    data_time = data_time[data_time.y==1].append(data_time_sample)
    print(time[i])
    print(data_time.shape)
    data_new = data_new.append(data_time)
```



### 查看缺失值

快速查看

```python
df.info()
df.isnull().sum().sort_values()
```

结构化缺失：某列缺失的位置，其他列也缺失，就是结构化缺失。

```python
# 如果某个pattern过多，考虑结构化缺失的可能性。
def row_miss(df):
    n_row, n_col = df.shape
    # 构建各行缺失pattern，e.g: '10001000110...'
    row_miss_check = df.applymap(lambda x: '1' if pd.isna(x) else '0')
    row_miss_pat = row_miss_check.apply(lambda x: ''.join(x.values), axis = 1)
    # 统计各pattern的频率
    pat_des = row_miss_pat.value_counts()
    print('The amount of missing patterns is {0}'.format(len(pat_des)))
    # 将各频率的累加表进行输出
    pat_des_df = pd.DataFrame(pat_des, columns=['row_cnt'])
    pat_des_df.index.name = 'miss_pattern'
    pat_des_df['cumsum_cover_pct'] = round(pat_des_df['row_cnt']/n_row, ndigits=2)
    
    if pat_des_df.shape[0]>10:
        print('The top 10 patterns info is:\n{0}'.format(pat_des_df.iloc[:10,:]))
    else:
        print('Patterns info is:\n{0}'.format(pat_des_df))
row_miss(df)

# 结构化缺失，应该探究原因，同步处理同时缺失的列，不能针对单独的一列缺失处理。
```



### 缺失值处理

##### 删除缺失值

```python
# 删特征列，缺失比例超过70%的特征，考虑直接删掉。
x_train = x_train.drop(['Cabin'],axis=1)
```

```python
# 删样本行，数据样本很多，缺失值很少，考虑删掉带缺失值的行
df = df.dropna(how='any')
```



##### 缺失值填充

###### fillna

```python
# 固定值填充
df = df.fillna(df.mean())

# 插值填充
train_data.fillna(method='pad', inplace=True) # 填充前一条数据的值，但是前一条也不一定有值
train_data.fillna(method='bfill', inplace=True) # 填充后一条数据的值，但是后一条也不一定有值
```



###### SimpleImputer

![1576119201911](https://raw.githubusercontent.com/zhanghongyang42/images/main/1576119201911-1603354042979.png)

```python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="constant",fill_value=0)

# 单列数据要做处理,符合输入格式。多列不用。
df["age"] = imp.fit_transform(df["age"].values.reshape(-1,1))
```



###### KNNImputer

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=2, weights="uniform")
imputer.fit_transform(X)
```



###### 时序插值

```
df.interpolate(method='time', inplace=True)
```



##### 缺失值有意义

缺失值不是单纯的缺失，而是有一定的业务含义。

```python
# 给缺失值赋值
给缺失值 赋值 一个不存在的值，比如 -1 或者 ‘nan’ 。赋值之后可以进行编码处理。
```

```python
# 缺失值二值化。或者手动二值化
from sklearn.impute import MissingIndicator

indicator = MissingIndicator(missing_values=-1)  #值为-1代表缺失
mask_missing_values_only = indicator.fit_transform(X)
```



### 查看异常值

##### 箱形图

上四分位数（Q3），中位数，下四分位数（Q1） ，四分位数差（IQR，interquartile range）Q3-Q1 

```python
sns.boxplot(x=i,data=df)
```



##### Isolation Forest



### 异常值处理

异常值大部分情况下都可以视为缺失值，可以进行 填充（替换），删除，有意义 等处理。



##### 箱型图删除

```python
# 定义一个上下限
lower = data['月销量'].quantile(0.25)-1.5*(data['月销量'].quantile(0.75)-data['月销量'].quantile(0.25))
upper = data['月销量'].quantile(0.25)+1.5*(data['月销量'].quantile(0.75)-data['月销量'].quantile(0.25))

# 过滤掉异常数据
qutlier_data=data[(data['月销量'] < lower) | (data['月销量'] > upper)]
```



##### 3σ原则删除

如果数据符合正态分布，可以用3σ原则删除数据。

```python
data['three_sigma'] = data['月销量'].transform( lambda x: (x.mean()-3*x.std()>x)|(x.mean()+3*x.std()<x))
correct_data=data[data['three_sigma']==False]
```



##### 异常值替换

业务中明显异常，又大量出现的某个值，直接替换。或者当作缺失值进行处理。

```python
df[col] = df[col].replace({365243: np.nan})
```



# 特征工程2

https://zhuanlan.zhihu.com/p/466685415

https://www.zhihu.com/column/c_1114217169679921152



### 数值化



##### LabelEncoder		

将类别型特征编码成 1，2，3，适用于有序特征（如1级、2级）

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train[col] = le.fit_transform(train[col])

le.inverse_transform(label)		#逆转
le.classes_						#类别数
```



##### OneHotEncoder

将类别型特征编码成 001，010，100，适用于无序特征（如 红黄绿），提高模型准确性和泛化能力

```python
from sklearn.preprocessing import OneHotEncoder

# sparse=True 输出稀疏的格式。
# handle_unknown='ignore'，忽略没fit到的值，编码00000
onehot = OneHotEncoder(sparse=False,handle_unknown='ignore')
onehot.fit_transform(df)
```



### 高基数编码

如果特征是无序的，比如25600种颜色，不适用于 LabelEncoder，OneHotEncoder 特征列又太多。可以采用以下高基数特征编码方法。

包括特征哈希和 TargetEncoder。

![1575356771847](https://raw.githubusercontent.com/zhanghongyang42/images/main/1575356771847-1603441279224.png)



##### 特征哈希

特征哈希把大规模高维稀疏特征转化为固定长度m的特征向量。m<原特征数量。



哈希表是一种数据结构。每个哈希表都是用一个哈希函数来实现键-值（key-value）对的映射。
哈希函数是任何一种可以将数据压缩成散列值的函数。理想的哈希函数会把不同的键散列到不同的块中。
但是大多数哈希表都存在哈希碰撞（hashing collision）的可能，即不同的键可能会被映射到相同的值上。

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/13838784-c5af13dbe03bfde1.png)

有了哈希表后，想要一个长度为5的 特征哈希 结果，‘the quick brown fox’ 对应的输出则是每个值出现的次数 [1,2,0,1,0]



```python
from sklearn.feature_extraction import FeatureHasher

h = FeatureHasher(n_features=10, input_type='dict')
D = [{'dog': 1, 'cat': 2, 'elephant': 4}, {'dog': 2, 'run': 5}]
h.transform(D).toarray()
```

```python
from sklearn.feature_extraction import FeatureHasher

h = FeatureHasher(n_features=10, input_type='pair')
D = [(('dog', 1), ('cat', 2)), (('dog', 2), ('run', 5))]
h.transform(D).toarray()
```

```python
from sklearn.feature_extraction import FeatureHasher

h = FeatureHasher(n_features=6, input_type='string')

hashed_df = pd.DataFrame(h.transform(df['col1'].astype(str)).toarray(), columns=[f'feature_{i}' for i in range(hashed_features.shape[1])])
df = pd.concat([hashed_df, df.drop(columns=['col1'])], axis=1)
```



##### Bin-counting

编码为一个实际值介于 0 和 1 之间的特征，用 该值下目标变量的多个条件概率 （当 x 为一个值时y 有多大可能为0，1）来代替一个特征，适用于树模型。

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/v2-ef3f363a83fd58142df1b101550d3ff8_720w.webp)

```python
class ConditionalProbabilityEncoder:
    def __init__(self, columns, label):
        self.columns = columns
        self.prob_dfs = {}
        self.label = label

    def fit(self, X):
        y = X[self.label].astype("category")

        for column in self.columns:
            X[column] = X[column].astype("category")

            prob_df = pd.DataFrame(index=X[column].cat.categories)
            for label in y.cat.categories:
                prob = (X[X[self.label] == label].groupby(column).size() / X.groupby(column).size())
                prob_df[f"P(label={label}|{column})"] = prob

            prob_df.fillna(0, inplace=True)
            self.prob_dfs[column] = prob_df

        return self

    def transform(self, X):
        for column in self.columns:
            X = X.join(self.prob_dfs[column], on=column, how="left")
        return X


encoder = ConditionalProbabilityEncoder(columns=["col1", "col2"], label="label")
encoder.fit(train_df)
train_df = encoder.transform(train_df)
test_df = encoder.transform(test_df)
```



### 离散化/分箱

主要对线性模型提升较大，对树模型影响较小。



##### 连续数据的离散化

等宽分箱：将区间等宽分割，可能出现空箱。

等频分箱：保证每个区间数据量一致，可能将原本是相同的两个数值却被分进了不同的区间。



###### KBinsDiscretizer

```python
from sklearn.preprocessing import KBinsDiscretizer

# strategy：uniform（等宽分箱）、quantile（等频分箱）、kmeans（聚类分箱）
est = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='uniform')

data = est.fit_transform(data)
```



###### cut、qcut

```python
# 等宽分箱
df['Fare'] = pd.cut(df['Fare'], 4, labels=range(4))

# 等频分箱
df['Fare'] = pd.qcut(df['Fare'],4,labels=['bad','medium','good','awesome'])

# 自定义分箱
bins=[0,200,1000,5000,10000]
df['mout_bin']=pd.cut(df['mout'],bins)
```



##### 连续数据的二值化

大于阈值为1，小于阈值为0

```python
from sklearn.preprocessing import Binarizer

bir = Binarizer(threshold=df['Fare'].mean())

df['Fare'] = bir.fit_transform(df['Fare'].values.reshape(-1,1))
```



##### 时间数据的离散化

时间数据提取星期，小时，月份等。



##### 离散数据的离散化

就是把很多小箱 变成几个大箱。



### 无量纲化

也称为 标准化、归一化 、中心化，但在翻译时经常被混用，所以这里我们统称为**无量纲化**或者特征缩放。

`无量纲化：将不同规格的数据转换到同一规格，或将不同分布的数据转换到某个特定分布。`



特征缩放方法：

1、特征值/特征最大值

2、mean normalization 均值归一化    (x-x均)/(最大-最小)

3、Z-score normalization  (x-x均)/标准差



![无量纲化](https://raw.githubusercontent.com/zhanghongyang42/images/main/无量纲化.PNG)



作用：1、加快速度（使用梯度下降的算法中，譬如逻辑回归，支持向量机，神经网络）

​			*线性回归中，如果两个特征的值范围相差很大，那么他们的参数范围相差也会很大，绘制出的代价函数和参数的等高线图趋于扁平，在这种代价函数上使用梯度下降训练不容易收敛。所以需要将特征进行缩放，加快训练速度*

​			2、统一量纲，提升精度（在距离类模型，譬如K近邻，K-Means聚类中）  

​			3、改变数据分布。



##### StandardScaler

z =（x - u）/ s ，减均值除标准差。

标准化 后效果：特征的均值为0，标准差为1。把数据变为标准正态分布。

适用于有outlier（测试集的数据范围超过或部分超过了训练集）数据的情况，适用于有较多异常值和噪音的情况。

```python
# 可以自动处理缺失值
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data)
x_std = scaler.transform(data)

scaler.inverse_transform(x_std) #逆转标准化

# 标准化前的均值与方差
print(scaler.mean_)
print(scaler.var_)	

# 标准化后的均值与方差
print(x_std.mean())	
print(x_std.std())	
```



##### MinMaxScaler

X_std = (X - X.min()) / (X.max() - X.min()) 。

归一化 后效果：特征值处于某特定范围，如（0，1）。

适用于无outlier数据的情况。在标准差较小的特征上表现较好。

```python
# 可以自动处理缺失值
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler = scaler.fit(data)
result = scaler.transform(data)

scaler.inverse_transform(result)   #将归一化后的结果逆转
```



##### MaxAbsScaler

X_std =  X / abs(X.max) 。

缩放 后效果 ：使特征处于（-1，1）的范围内。

不会破坏稀疏性，0仍然是0。

```python
from sklearn.preprocessing import MaxAbsScaler

max_abs_scaler = MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
```



##### RobustScaler

$$
x' = \frac{x - \text{median}(X)}{\text{IQR}(X)}
$$

减去中位数，除以（四分之三位数-四分之一位数）

在异常值多的情况下使用。

```python
from sklearn.preprocessing import RobustScaler

transformer = RobustScaler().fit(X)
transformer.transform(X)
```



### 特征衍生

##### PolynomialFeatures

数值型特征相乘，类别型特征拼接

```python
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# 构造示例 DataFrame，包含3个特征和1个标签
data = {
    'feature1': [1, 2, 3, 4],
    'feature2': [5, 6, 7, 8],
    'feature3': [9, 10, 11, 12],
    'label':    [0, 1, 0, 1]
}
df = pd.DataFrame(data)

# 分离特征和标签
features = df[['feature1', 'feature2', 'feature3']]
label = df['label']

# 创建 PolynomialFeatures 对象，degree=2 表示二次多项式；不包含截距项避免重复原始特征中的常数项
poly = PolynomialFeatures(degree=2, include_bias=False)

# 对特征进行多项式特征转换
poly_features = poly.fit_transform(features)

# 获取转换后特征的名称
poly_feature_names = poly.get_feature_names_out(features.columns)

# 构造新的 DataFrame，包含所有多项式特征
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

# 合并原始特征、多项式衍生特征和标签
df_new = pd.concat([features, df_poly, label], axis=1)

print(df_new)
```



##### 专家经验构建特征

1、领域知识特征

2、交互特征：特征间进行加减乘除，同比/环比 ，差分等数学运算。

3、聚合特征：通过统计方法（如均值、方差、最大值、最小值等）构造汇总性特征。 3个月点击 或者 上月点击。

4、时序或周期性特征：趋势、季节性、周期性以及滞后特征

5、字符串特征：从字符串或者身份证中切割提取特征。



### 特征选择

特征选择包括 3种类型的方法：

​	1.**Filter**：根据自变量和目标变量的关联来选择变量。包括**方差选择法** 和利用相关性、卡方、互信息等指标的**单变量特征选择法**。

​	2.**Wrapper**：利用模型`coef_` 属性 或 `feature_importances_` 属性获得变量重要性，排除特征，一个模型多次迭代的**递归特征消除法**。

​	3.**Embedded**：利用模型单独计算出每个特征和Lable的系数，设置阈值，删除特征。一次训练删除多个特征的**SelectFromModel**。



实际使用：方差选择法做预处理，然后使用递归特征消除法



##### 方差选择法

x拔是期望，n是总数。

![b360e15e91954d8a4cf7521a9190d888](https://raw.githubusercontent.com/zhanghongyang42/images/main/b360e15e91954d8a4cf7521a9190d888.svg)



计算每一列特征的方差，小于给定方差阈值的特征即删除。

方差越小代表该列没有变化，也不能反映y的变化情况。

```python
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# 构造示例 DataFrame
data = {
    'A': [0, 1, 0, 1],
    'B': [2, 2, 2, 2],  # 方差为0的常量特征
    'C': [5, 6, 7, 8]
}
df = pd.DataFrame(data)

# 初始化 VarianceThreshold，设定阈值为0.0（移除所有方差为0的特征）
selector = VarianceThreshold(threshold=0.0)
df_selected_array = selector.fit_transform(df)

# 获取保留特征的名称
selected_columns = df.columns[selector.get_support()]

# 将结果转换为 DataFrame
df_selected = pd.DataFrame(df_selected_array, columns=selected_columns)

print(df_selected)
```



##### 单变量特征选择

根据自变量和目标变量的关联来选择变量。关联指标包括相关系数、卡方、互信息。

单变量特征选择准确性一般，但相比其他模型选择方法速度更快。不推荐使用。



SelectBest 只保留 k 个最高分的特征，

SelectPercentile 只保留百分比最高分的特征。

```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

selector = SelectKBest(chi2, k=2).fit(X, y)
X_new = selector.transform(X)

# 相似性得分越大越好，pvalue越小越好
print(selector.scores_)
print(selector.pvalues_)
print(selector.get_support(True))
```

注：稀疏格式数据只能使用卡方检验和互信息法



指标

```
分类指标：
	chi2，卡方检验
	f_classif，方差分析
	mutual_info_classif，互信息
回归指标：
	f_regression，皮尔逊相关系数
	mutual_info_regression：最大信息系数MIC
```



###### 卡方检验原理

卡方值越小，越代表特征中的每个值对label的影响差不多，即这个特征与label关系不大。

![6342708-31a2138c73d6431e](https://raw.githubusercontent.com/zhanghongyang42/images/main/6342708-31a2138c73d6431e.webp)



理论频次的计算示例

![6342708-f1d3e1b69de0c74a](https://raw.githubusercontent.com/zhanghongyang42/images/main/6342708-f1d3e1b69de0c74a.webp)

![6342708-59bd9a56f3edd31f](https://raw.githubusercontent.com/zhanghongyang42/images/main/6342708-59bd9a56f3edd31f.webp)



##### 递归特征消除法

Recursive Feature Elimination （RFE）通过`coef_` 属性 或者 `feature_importances_` 属性获得重要性，在越来越小的特征集合上递归的选择特征进行删除。

```python
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)  #n_features_to_select：剩几个，step：每次删几个

df = rfe.fit_transform(X, y)
```



RFECV ：每次选择不同列进行RFE 完成交叉验证，找到最优特征。

```python
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),scoring='accuracy')

rfecv.fit(X, y)
x = rfecv.transform(X)

print("Optimal number of features : %d" % rfecv.n_features_)
print("Ranking of features : %s" % rfecv.ranking_)
print("Support is %s" % rfecv.support_)
print("Grid Scores %s" % rfecv.grid_scores_)
```



##### SelectFromModel

`SelectFromModel` 是一个 meta-transformer（元转换器） 。它可以用来处理任何带有 `coef_` 或者 `feature_importances_` 属性的训练之后的评估器。 

如果相关的`coef_` 或者 `feature_importances_`  属性值低于预先设置的阈值，这些特征将会被认为不重要并且移除掉。



###### 基于树的特征选择

树模型可以找到非线性关系，但是有两个本身的问题：重要特征没被选择出来，对类别多的变量有利。

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
model = SelectFromModel(clf, prefit=True)
X_new = model.fit_transform(X)
```



##### 变量聚类特征选择

```python
# 给变量进行聚类，每类中有若干个变量
from varclushi import VarClusHi
vc=VarClusHi(trainData[var_list],maxeigval2=0.6,maxclus=None)

# N_Vars变量数量,Eigval1方差
print(vc.info)

# 查看分类结果
vc_list=vc.rsquare.reset_index()
vc_list=pd.DataFrame(vc_list,columns=['Cluster','Variable','RS_Own'])
print(vc_list)

# 随机森林训练重要性
X = trainData[var_list]
y = trainData[target]

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(random_state=42)
RFC_Model = RFC.fit(X,y)

importances = []
for i in vc_list['Variable']:
	importances.append(RFC_Model.feature_importances_[i])
vc_list['importances'] = importances

# 根据聚类结果及重要性进行筛选
vc_list = vc_list.groupby('Cluster').apply(lambda x:x[x['importances']==x['importances'].max()])
var_list = list(vc_list[vc_list['Importances'] >= 0.0001].Variable)
```



### 降维处理

特征选择之后，仍有大量特征，可选择降维算法



### 自定义特征处理

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# 传入函数，进行包装
transformer = FunctionTransformer(np.log1p)

transformer.transform(X)
```



### Silver Bullet

##### category 频度统计

对GBDT十分有效。原理：干掉刺头，长尾的值全部变为他的频度，使学习难度降低。

```python
df_counts = df['区域'].value_counts().reset_index()
df_counts.columns = ['区域','区域频度统计']
df = df.merge(df_counts,on =['区域'],how='left')
```



# 数据划分

划分训练集和测试集，训练集用于模型训练，测试集用于模型评价。

划分特征 和 label。

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```



# 模型概述

机器学习模型分为**监督学习**，**半监督学习**，**无监督学习**，**强化学习** 等。

监督学习包括 **分类模型**，**回归模型**。很多模型都既可以做分类，又可以做回归。

无监督学习包括 **聚类模型**，**降维模型**，**异常检测模型**等。



**多标签/多输出**：监督学习 根据输出的不同，可以有 **回归、多输出回归、二分类、多分类、多输出二分类（多标签分类）、多输出多分类（多任务分类）**。

**集成模型**：可以多个模型（一般为监督学习模型）融合成一个模型，根据融合方式的不同，分为 **bagging、boosting、stacking**  。



该文档涉及到的所有算法都还需要：一、手动推导公式。二、使用代码实现（查看吴恩达代码示例）



# 监督学习



## 线性回归



### 单变量线性回归

Linear Regression with One Variable



##### 模型表示

![image-20220727134254238](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220727134254238.png)



##### 代价函数

代价(损失)函数：损失函数是衡量模型预测值与真实值之间误差的数学指标，通过最小化该误差来求得模型参数。

目标函数：损失函数+正则项



损失函数描述**经验风险**（预测值与真实值误差越大，经验风险越高）。

正则项描述**结构风险**（即过拟合风险或者函数复杂度高风险），正则化使得函数变简单，降低了结构风险。





线性回归的代价函数是平方误差代价函数。

![image-20220727135152324](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220727135152324.png)

代价函数图像：一图是x和y的图，二图是w和 J 的图，三图是 w与b 和 J 的图。

![image-20220801171113352](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220801171113352.png)



##### 梯度下降

梯度下降：一个用来求代价函数最小值的算法，我们将使用梯度下降算法来求出代价函数的最小值。



梯度下降背后的思想是：开始时我们随机选择一个参数的组合，计算代价函数，然后我们寻找下一个能让代价函数值下降最多的参数组合。我们持续这么做直到到到一个局部最小值（**local minimum**），因为我们并没有尝试完所有的参数组合，所以不能确定我们得到的局部最小值是否便是全局最小值（**global minimum**），选择不同的初始参数组合，可能会找到不同的局部最小值。

梯度下降 结合 凸函数具有唯一的一个全局最小值，*（证明省略）*，即可得到全局最小值。





按照以下公式进行梯度下降，多次更新参数，直到导数为0的时候，证明到达了代价函数局部最小点。梯度下降完成。

w、b 为要计算的参数， α 为步长 。



![image-20220727190217157](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220727190217157.png)

偏导数是在多变量函数中，当固定其他变量不变时，对某一变量求导所得到的导数。即代价函数对w和b求导，就是两个偏导数。

梯度是函数在某点上所有偏导数组成的向量，它指示了函数值上升最快的方向，而负梯度方向则是下降最快的方向*（证明省略）*。

所有参数同步更新，即为梯度下降法。





单变量线性回归的梯度下降公式如下

![image-20221229151253984](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20221229151253984.png)



批量梯度下降：是指在每次参数更新时，利用整个训练集来计算代价函数的梯度，从而保证更新方向的准确性，但计算量随数据规模增大而显著增加。

随机梯度下降：用一个随机训练样本更新梯度。

小批量梯度下降：用部分训练集更新梯度。



##### 梯度下降收敛

通过画迭代次数和代价函数的图，可以找出梯度下降接近收敛的最佳迭代次数。

如果代价函数没有一直下降，说明学习率α的选择有问题。

![image-20220801172806650](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220801172806650.png)



##### 学习率设置

观察上图，代价函数随着迭代次数的增加，反复波动，或者反而一直增大，一般都是因为学习率过大。

α 过小，计算很慢，α 过大，可能导致函数不收敛甚至发散，就是直接跨过了函数最低点。

![image-20220801184731842](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220801184731842.png)

选择合适的学习率：

可以把学习率设置为0.0001，0.001，0.01，0.1等，只迭代几次，观察代价函数的变化。



### 多变量线性回归



![image-20220728093246113](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220728093246113.png)



### 线性回归代码

```python
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

print(reg.coef_)
```



## 逻辑回归

可以实现二分类的算法。



### 模型表示

逻辑回归：线性回归 + sigmoid函数

![image-20220802104359451](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220802104359451.png)



### 决策边界

决策边界就是希望能把样本正确分类的一条边界，主要有线性决策边界(linear decision boundaries)和非线性决策边界(non-linear decision boundaries)。

画出决策边界可以可视化的看出分类效果。

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/1355387-20180726184054400-1134632155.png)

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/1355387-20180726190653105-1761881987.png)

每一个轴对应特征x，不同颜色代表不同的分类/y。

当一个模型的参数确定，分类阈值确定，即可画出决策边界。

如在逻辑回归中，参数确定 y=g(3+x1+x2) 。分类标准确定，y>0.5为1类，y<0.5为0类。那么当概率为0.5，即得到逻辑回归的决策边界公式3+x1+x2=0。



### 代价函数



##### 引言

如果和线性回归一样，采用平方误差作为逻辑回归的代价函数是不好的。

如果采用平方误差作为逻辑回归的代价函数，则代价函数不是凸函数，采用梯度下降求代价函数最小，会有很多的局部最小值作为干扰。

![image-20220802155953402](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220802155953402.png)



损失函数的构造思路：即找到真实值 y 和预测值 f 的差距 。

逻辑回归的真实值是 0或1，预测值为 01之间 的数。



##### 直观理解

因为 逻辑回归的真实值是 0或1 ，所以我们分开构造损失函数。

当真实标签是1，我们希望预测值越接近1，损失函数越小；预测值为1时，损失函数为0；预测值越接近0，损失函数无限大，这符合sigmod的特点。

当真实标签是0，同上。

下面两图通过 log 函数实现了上面的需求，横轴为预测值，纵轴为损失函数。

![image-20220802162258108](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220802162258108.png)

![image-20220802162124359](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220802162124359.png)

上面两图可以合并为交叉熵损失公式。

逻辑回归采用交叉熵损失函数

![math](https://raw.githubusercontent.com/zhanghongyang42/images/main/math.svg)



##### 公式推导

交叉熵损失推导（伯努利分布的极大似然估计）：

​	1.逻辑回归函数的结果y满足**伯努利分布**：即 y=1的概率 和 y=0的概率 相加为1。

​		y\_pred代表y=1的预测值。
$$
y\_pred=P(y=1|x)
$$

$$
1-y\_pred=P(y=0|x)
$$

​	2.合并上面两个公式，单个样本预测正确的概率可以写成 
$$
P(y|x)=y\_pred*y + (1-y\_pred)^(1-y)
$$
​	3.所有样本预测正确概率就是对上式概率连乘。（上面23步就组成一个似然函数。极大似然说明模型拟合的好。不用理解这句话）

​	5.对极大似然函数取log，即为交叉熵损失。加负号，方便优化。



### 梯度下降

梯度下降公式推导

![image-20221229165704733](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20221229165704733.png)



### 逻辑回归代码 

是**二分类**算法。已经封装**多分类**的改进。也可以用分类概率值进行**排序**。

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X, y)

clf.predict(X)
clf.predict_proba(X)
clf.score(X, y)
```



| 参数         | 默认值 | 解释                                                         |
| ------------ | ------ | ------------------------------------------------------------ |
| solver       | lbfgs  | 求解代价函数算法。                      <br />小数据：‘liblinear’                         <br />中数据：‘newton-cg’，‘lbfgs’       <br />大数据：‘sag’，‘saga’ |
| max_iter     | 10     | 最大迭代次数，仅在求解代价函数算法为 newton-cg,、sag、lbfgs 才有用 |
| tol          | 0.0001 | 训练时停止求解的标准                                         |
|              |        |                                                              |
| penalty      | l2     | 正则化。 与求解代价函数算法有关<br />‘liblinear’：    ‘l1’，‘l2’ <br />‘newton-cg’：‘l2’，‘none’<br />‘lbfgs’：          ‘l2’，‘none’<br />‘sag’：            ‘l2’，‘none’ <br />‘saga’：          ‘elasticnet’，‘l1’，‘l2’，‘none’ |
| C            | 1.0    | 正则化系数λ的倒数，越小的数值表示越强的正则化                |
|              |        |                                                              |
| multi_class  | auto   | 多分类的方式：‘auto’， ‘ovr’， ‘multinomial’                 |
| warm_start   | False  | 热启动，使用上一次训练结果继续训练                           |
| random_state | None   | 随机数种子                                                   |



逻辑回归的求解算法优缺点分析：

1. **liblinear**：适用于稀疏数据，易陷局部最优，高维特征时计算较慢。
2. **newton-cg**：收敛快、迭代少，大数据集下计算较慢。
3. **lbfgs**：大数据集下可能因内存占用过高而受限。
4. **sag**：专为大规模数据设计，收敛快但稳定性略差。
5. **saga**：相比sag，能进一步加快收敛，但相对消耗更多内存。



不同情况下应该选择的求解算法

| 多分类 + 正则化                  | `liblinear` | `lbfgs` | `newton-cg` | `sag` | `saga` |
| -------------------------------- | ----------- | ------- | ----------- | ----- | ------ |
| multinomial + L2罚项             | ×           | √       | √           | √     | √      |
| 一对剩余（One vs Rest） + L2罚项 | √           | √       | √           | √     | √      |
| multinomial + L1罚项             | ×           | ×       | ×           | ×     | √      |
| 一对剩余（One vs Rest） + L1罚项 | √           | ×       | ×           | ×     | √      |
| elasticnet                       | ×           | ×       | ×           | ×     | √      |
| 无罚项                           | ×           | √       | √           | √     | √      |
| **表现**                         |             |         |             |       |        |
| 大数据集上速度快                 | ×           | ×       | ×           | √     | √      |
| 未缩放数据集上鲁棒               | √           | √       | √           | ×     | ×      |



## 多分类算法

使用一对多 (**One vs Rest**) 或者一对一**（One vs One）**把多分类问题转化为二分类问题解决。



### One vs Rest

基本思想：n 种类别的样本进行分类时，取其中一个类别的样本作为正类，将剩余的所有类别的样本看做父类，这样就形成了 n 个二分类问题。使用二分类算法训练出 n 个模型，每个模型都预测出该类别正类的概率，所得概率最高的类别即认为是该预测样本的类别。



### One vs One

给定一个具有n个类别的多分类问题，OvO策略会为每一对类别训练一个二分类模型，因此总共需要训练 C(n, 2) 个模型，其中 C(n, 2) = n(n-1)/2 是从n个类别中选择2个的组合数。

在训练每个二分类模型时，我们只使用属于这两个类别的数据样本。例如，对于类别i和类别j的二分类模型，我们只使用标签为i或j的样本来训练这个模型。



在预测时，我们将输入特征提供给所有训练好的二分类模型。每个模型会输出一个预测结果，即它认为输入特征属于哪个类别。最后，我们统计所有模型的投票结果，得票最多的类别就是最终预测结果。



OvO算法的优点是每个模型只需要在一部分数据上进行训练，这样可能使得训练过程更简单、容易收敛。但是，由于需要训练的模型数量随着类别数量的增加而快速增加，因此在类别数量较多时，OvO策略的计算成本可能较高。



## ---- ---- ---- ----



## 树模型

### 信息论

##### 信息熵

信息熵香农提出，从物理中熵演化而来。

物理学中“熵”是指物体能量的分布更加均匀，即熵越大，代表状态越混沌。

"信息熵”可以用概率表示，概率越大，越确定，越不混沌，"信息熵”越小。



计算一个事件的信息熵：

一个事件有多个结果，如果这些结果相互独立，则此事件的信息熵为各个事件的不确定性之和，即每个独立事件发生概率的倒数之和，为了满足可加性，使用log

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/clip_image002.gif)



##### 信息增益

我们使用信息增益来进行特征选择。信息增益越大，说明该特征越能确定结果。

整体数据集的信息熵（I(parent)）一般是保持不变的，信息增益就是（I(parent)）- 当前节点的信息熵。

​    ![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/clip_image002-1642667703064.gif)

信息增益计算示例：

![image-20220120165538960](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220120165538960.png)

![image-20220120165545082](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220120165545082.png)



##### 信息增益率

信息增益率克服了信息增益偏向选择取值多的属性的不足。

 信息增益率：Gainr(A) = Info(D)-Info_A(D) / H(A) ：其中H(A)为A的熵



### 决策树原理

ID3	二分类、多分类

C4.5	二分类、多分类

cart	二分类、多分类、回归



##### 决策树介绍

决策树算法是一种监督学习算法。生成好的决策树如下：

- 每个内部节点是通过一个特征值 划分某个特征

- 每个叶节点代表一种分类结果。

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/Wed%2C%2014%20Jun%202023%20185520.jpeg)



**构建决策树三个步骤**：特征选择、决策树生成、决策树剪枝

**特征选择的指标：**信息增益，信息增益率，基尼系数，平方误差和。

**构造决策树算法：**

- ID3算法：使用信息增益进行特征选择。进行分类。
- C4.5算法：使用信息增益率进行特征选择。进行分类。
- CART算法：使用基尼系数和平方误差和进行特征选择。用于分类和回归。

**决策树剪枝：**为了防止过拟合，可以对决策树进行剪枝。

| 预剪枝                                                       | 后剪枝                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 决策树生成过程中，节点划分前估计，如果不能提升，则标为叶子节点。 | 决策树生成后，自底向上对非叶子节点估计，将子树节点变为叶子节点。 |
| 利用树结构进行限制                                           | 通过测试集结果进行限制                                       |
| 可以降低过拟合的风险而且还可以减少训练时间，但另一方面它是基于“贪心”策略，会带来欠拟合风险。 | 欠拟合风险很小，泛化性能往往优于预剪枝决策树。但同时其训练时间会大的多。 |



##### ID3算法

###### 构建过程

根据信息增益来选择进行划分的特征，然后递归地构建决策树。

1. 从根节点开始，计算所有可能的特征的信息增益，选择信息增益最大的特征作为节点的划分特征；
2. 由该特征的全部不同取值建立子节点。
3. 再对子节点递归1-2步，构建决策树；
4. 直到没有特征可以选择或类别完全相同为止，得到最终的决策树。
5. 叶子节点中哪个类别多，所有这个叶子节点的数据就算做哪个类别。



###### 缺点

1、没有剪枝，可能会过拟合。

2、特征选择使用信息增益，会偏向那些取值较多的特征。

3、只能处理离散分布的特征，做**分类任务**。因为使用信息熵，只能分类。

4、没有处理缺失值。



##### C4.5算法

C4.5是对ID3的改进，包括以下方面：

- 使用信息增益率来代替信息增益选择属性。
- 连续特征离散化。
- 处理缺失值。
- 引入剪枝策略。



###### 连续特征离散化

c4.5 连续特征划分时，最终会划分成二叉树结构，离散特征划分时，同id3一样，是多叉树结构。

将连续特征离散化，假设 n 个样本的连续特征 A 有 m 个取值，C4.5 将其排序并取相邻两样本值的平均数共 m-1 个划分点，计算以各个划分点划分成二叉树的信息增益率，选择信息增益率最大的那个划分点。



###### 缺失值的处理

一是在样本某些特征缺失的情况下选择划分的属性。

二是选定了划分属性，对于在该属性上缺失特征的样本的处理。



对于第一个子问题，某一个有缺失特征值的特征A。

C4.5的思路是将数据分成两部分，一部分是有特征值A的数据D1，另一部分是缺失特征A的数据D2。

根据D1数据计算特征A的各种指标，信息增益或者信息增益率。然后乘上一个系数，这个系数是A特征无缺失的样本占总样本的比例。



对于第二个子问题

可以将缺失特征的样本同时划分入所有的子节点，不过将该样本的权重按各个子节点样本的数量比例来分配。

即 特征A有3个特征值A1,A2,A3 ，数量为2,3,4，缺失值分别划入A1,A2,A3，划入A1时，缺失值的值是2/9*A1，以此类推。



###### 剪枝处理

**预剪枝（Pre-pruning）**

算法在构建决策树的过程中，对每个节点进行判断，若满足剪枝条件，则停止分裂，将该节点标记为叶节点，并赋予该节点该节点所属的最常出现的类别。

1. 限制树的最大深度、节点数或样本数。
2. 限制分裂到某一层时所需要的最小信息增益或最小样本数。
3. 限制叶节点中样本数最小或占比最小。



**后剪枝（Post-pruning）**

算法先用已经构建好的完整决策树，对测试集进行预测，记录预测错误的样本数。

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/v2-b59938e00ae27529e00ffa9a2a7c1c81_720w.webp)

从最底层的叶节点开始，反向遍历决策树，在每个节点，合并一次叶子节点，然后重新预测测试集看测试集的预测性能有没有提升，有就剪枝——也就是合并叶子节点，没有就不合并。这种剪枝的方法显然计算开销太大了，数据量大的情况下压根没法用，而且明显容易过拟合测试集。

在后剪枝的过程中，可以使用交叉验证的方法来选择最优剪枝点，即将数据集划分成若干份，每次使用其中一个子集作为测试集，计算预测效果，选择最优的剪枝点。这种方式能够更好地避免剪枝后模型过拟合的问题。



而c4.5的悲观剪枝（PEP）又不同于普通的后剪枝（也叫REP(错误率降低修剪)），不使用测试集，直接在训练集上进行剪枝操作。

https://zhuanlan.zhihu.com/p/86679767



###### 缺点

- 剪枝策略可以再优化；
- C4.5 用的是多叉树，用二叉树效率更高；
- C4.5 **只能用于分类**；



##### cart算法

###### 改进点

- ID3 和 C4.5 只可以处理分类问题，cart 既可以处理分类也可以处理回归。cart分类时使用基尼系数，回归时使用平方误差和。


- ID3 和 C4.5 是多叉树，cart 是二叉树。
- 连续的特征离散化。区别在选择划分点时，C4.5是信息增益率，CART是基尼系数。

- CART 采用surrogate splits（代理特征分裂）来估计缺失值，而 C4.5 以不同概率划分到不同节点中。
- CART 采用“基于代价复杂度剪枝”方法进行剪枝，而 C4.5 采用悲观剪枝方法。



###### 基尼系数

如果我们的数据集中有K个分类，一个数据点正好是第k个分类的概率是p(k)，基尼系数公式如下：

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/v2-82f818b42b55ff6490bcda9d9cad7182_b.webp)

举例计算如下：

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/1947150-20200527060046089-1867013542.png)

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/1947150-20200527065831679-1635125971.png)

基尼系数也称为基尼不纯度。基尼系数越小，表示数据集越纯。直到纯到全部一样，就是叶子节点了。



###### 平方误差和

$$
SSE = ∑i∈D (y_i - ŷ_i)^2
$$

使用平方误差和做回归时，叶子节点的值是训练是落到这个叶子节点所有数据的标签平均数。



###### 离散特征的处理

id3和C4.5都是将离散特征的每一个值划分为多叉树。

cart 是不停的二分离散特征。且一个特征可能会参与多次节点的建立。

cart把所有的构成二叉树的情况都计算出基尼系数，CART会考虑把特征A分成A1 和 A2,A3、A2和A1,A3、A3和A1,A2 三种情况，找到基尼系数最小的组合，如 A2和A1，A3。于这次没有把特征A的取值完全分开，后面还有机会对子节点继续选择特征A划分 A1和A3。



###### 缺失值处理

对于缺失值，cart和c4.5的处理方式类似。

缺失特征分为缺失值和非缺失值两部分，在非缺失值部分正常计算gini，然后乘以缺失值所占比例，也其他特征比较。

如果缺失特征恰好是gini增益最大的特征，那么在这个缺失特征上分裂就比较麻烦。cart使用的方式是：surrogate splits（不用看），有两种情况：

1、首先，如果某个存在缺失值的特征恰好是当前的分裂增益最大的特征，那么我们需要遍历剩余的特征，剩余的特征中如果有也存在缺失值的特征，那么这些特征忽略，仅仅在完全没有缺失值的特征上进行选择，我们选择其中能够与最佳增益的缺失特征分裂之后增益最接近的特征进行分裂。

2、如果我们事先设置了一定的标准仅仅选择仅仅选择差异性在一定范围内的特征作为代理特征进行分裂而导致了没有特征和最佳缺失特征的差异性满足要求，或者所有特征都存在缺失值的情况下，缺失样本默认进入个数最大的叶子节点。



显然这种缺失值的处理方式的计算量是非常大的，而带来的性能提升确很有限，

sklearn中的树算法包括cart、gbdt、rf都没有提供对缺失值的处理功能。

**xgb和lgb处理缺失值**，先按非缺失值，计算出分裂点，再把缺失值分别填入左右，看填左右哪边增益大，就分到哪边。

这样在处理上要快速的多，而且在gbm的框架下一点点的误差其实影响不大。



###### 剪枝处理

在cart中使用基于代价复杂度剪枝，就是c4.5中最简单的那种剪枝技术。



##### 特征重要性

sklearn中是使用基尼指数来计算的。



思路：每个特征影响样本走向数量 占总样本数量 的比例 * 基尼系数减少的程度 = 特征重要性

具体公式：

```text
N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)

其中，N是样本的总数，N_t是当前节点的样本数目，N_t_L是结点左孩子的样本数目，N_t_R是结点右孩子的样本数目。impurity为基尼不纯度
```



举例：假设我们得到一个三个特征训练好的决策树如下：

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/v2-1b94eaf45163212c25bc7f94314966bd_b.webp)

特征重要性计算

```text
X0 的 feature_importance = (2 / 4) * (0.5) = 0.25
X1 的 feature_importance = (3 / 4) * (0.444 - (2 / 3 * 0.5)) = 0.083
X2的feature_importance = (4 / 4) * (0.375 - (3 / 4 * 0.444)) = 0.042
```

上述三个值加起来不是1，所以我们再归一化就可以了。



### 决策树代码

```python
from sklearn import tree

# 二分类，多分类，排序
clf = tree.DecisionTreeClassifier()

# 回归
clf = tree.DecisionTreeRegressor()

# 训练
clf = clf.fit(X, Y)

# 预测
clf.predict(X)
clf.predict_proba(X)
```

```python
# 分类模型的类别
clf.classes_
# 类别数量
clf.n_classes_

# 特征数量
clf.n_features_
clf.max_features_

# 输出结果数量
clf.n_outputs_

# 特征重要性
clf.feature_importances_
```

```
max_depth：决策树最大深度。一般为3-6。

max_features：特征切分时考虑的最大特征数量，默认是对所有特征进行切分。也可以制定具体的特征个数或者数量的百分比。
splitter:特征切分点选择标准，有“best”和“random”两种参数可以选择，best表示在所有特征上切分，random表示随机选择一部分特征，适用于数据集较大的时候。
criterion：特征选择的标准。使用信息增益的是ID3，使用信息增益比的是C4.5算法，使用基尼系数的CART算法。默认是gini系数。

max_leaf_nodes: 最大叶子节点数，默认是”None”，即不限制最大的叶子节点数。
min_samples_split:子数据集再切分需要的最小样本量，默认是2，子数据样本量小于2时，则不再进行切分。如果数据量较大，把这个值增大，限制子数据集的切分次数。
min_impurity_split:切分点最小不纯度，用来限制决策树的生成，如果某个节点的不纯度（可以理解为分类错误率）小于这个阈值，那么该点将不再进行切分。

min_samples_leaf:叶节点（子数据集）最小样本数，如果子数据集中的样本数小于这个值，那么该叶节点和其兄弟节点都会被剪枝（去掉），该值默认为1。
min_impurity_decrease:切分点不纯度最小减少程度，如果某个结点的不纯度减少小于这个值，那么该切分点就会被移除。

random_state:随机种子的设置。
min_weight_fraction_leaf:在叶节点处的所有输入样本权重总和的最小加权分数，如果不输入则表示所有的叶节点的权重是一致的。
class_weight:权重设置，主要是用于处理不平衡样本，与LR模型中的参数一致，可以自定义类别权重，也可以直接使用balanced参数值进行不平衡样本处理。
presort:是否进行预排序，默认是False，所谓预排序就是提前对特征进行排序，如果不进行预排序，则会在每次分割的时候需要重新把所有特征进行计算比较一次。
```



## 集成模型原理

ensemble 集成模型

**Bagging**：各个模型并行跑出结果后，进行投票（硬投票）或计算平均概率（软投票）得到最终结果。

**Boosting**：各个模型串行跑出结果进行模型效果提升。

**Stacking**：Bagging和Boosting混合使用。



### Bagging原理

Bootstrap aggregating ，简称 Bagging

Bootstrap 抽样方法：每个基学习器都会对训练集进行有放回抽样得到子训练集，比较著名的采样法为 0.632 自助法。每个基学习器基于不同子训练集进行训练。

aggregating ：综合所有基学习器的预测值进行投票（硬投票）或计算平均概率（软投票）得到最终结果。

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/v2-a0a3cb02f629f3db360fc68b4c2153c0_r.jpg)



### Boosting 原理

Boosting 训练过程为阶梯状，基模型的训练是有顺序的，每个基模型都会在前一个基模型学习的基础上进行学习，最终综合所有基模型的预测值产生最终的预测结果，用的比较多的综合方式为加权法。

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/v2-3aab53d50ab65e11ad3c9e3decf895c2_r.jpg)



### Stacking原理

Stacking 是先用全部数据训练好基模型，然后每个基模型都对每个训练样本进行的预测，其预测值将作为训练样本的特征值，最终会得到新的训练样本，然后基于新的训练样本进行训练得到模型，然后得到最终预测结果。类似于Bagging，只是最后不是使用投票，而是模型预测。

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/v2-f6787a16c23950d129a7927269d5352a_b.jpg)



### 效果提升原因

为什么集成模型可以提升效果。



我们可以使用模型的偏差和方差来近似描述模型的准确度。想要**提升准确度**，就得**降低偏差或者方差**。

不同集成模型，有的降低方差，有的降低偏差，原因如下表。

|          | 偏差                                                         | 方差                                                         |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Bagging  | 每个基模型的权重等于 1/m 且期望近似相等，整体模型的期望等于基模型的期望，整体模型的偏差和基模型的偏差近似。 | 在一定范围内，随着基模型数量增多，结果多样性增多，整体模型的方差减少。 |
| Boosting | Boosting 框架中采用基于贪心策略的前向加法，整体模型的期望由基模型的期望累加而成，所以随着基模型数的增多，整体模型的期望值增加，偏差变小。 | 每个基模型降低了在前一个模型的基础上降低了偏差，但是并没有哪个操作降低了基模型的方差，所以最后所有基模型加权后的结果方差也不会降低。 |

**Bagging 降低了方差。Boosting 降低了偏差。Stacking 类似于Bagging。**



为什么集成模型可以提升效果：

1、降低了方差或者偏差。提高泛化能力，提高预测准确性。

2、集成了不同模型的优势。注意，集成时是需要模型差异性的，融合的模型要具有各自“擅长”的预测样本，否则可能比单一的模型中最好的那个还要差。

3、使得模型更复杂，更好的拟合复杂数据的分布。



### 基模型选择

通过上一节内容，我们知道，Bagging 降低了方差。Boosting 降低了偏差。Stacking 类似于Bagging，降低了方差。

而弱模型是偏差高（在训练集上准确度低）方差小（防止过拟合能力强）的模型。

强模型是偏差低（在训练集上准确度高）方差大（防止过拟合能力弱）的模型。



集成模型和基模型互相补充，就能取得更好的效果。

所以，**Bagging 和 Stacking** 中的基模型选择**强模型（偏差低，方差高）**，强模型本身偏差低，集成后降低了方差。

**Boosting** 中的基模型选择**弱模型（偏差高，方差低）**，弱模型本身方差低，集成后降低了偏差。



## 集成模型代码

##### Bagging

相同模型并行训练，属于bagging

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

# 抽取随机子集的随机特征来构建多个估计器，能减小BaggingClassifier的方差，模型更稳定，但是可能会损失信息，加大偏差
bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
```

```python
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
```



##### Voting 

不同模型并行训练，属于bagging

```python
# 分类
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

# 硬投票
eclf=VotingClassifier(estimators=[('lr', clf1),('rf', clf2),('gnb', clf3)],voting='hard')
                      
# 软投票
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],voting='soft', weights=[2, 1, 2])
```

```python
# 回归
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor

reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
reg3 = LinearRegression()

ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
ereg = ereg.fit(X, y)
```



##### stacking

把所有模型的预测结果变成一个样本，用xgb进行再预测

```python
import numpy as np
from sklearn.model_selection import KFold

#看不懂就自己写画矩阵
def get_stacking(clf, x_train, y_train, x_test, n_folds=10):
    """这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .如果输入为pandas的DataFrame类型则会报错"""

    train_num, test_num = x_train.shape[0], x_test.shape[0]

    #这个矩阵用来存储所有验证集（即训练集）结果
    second_train = np.zeros((train_num,))
    #这个矩阵用来存储测试集（n个模型的预测均值）结果
    second_test = np.zeros((test_num,))
    # 这个矩阵用来存储测试集（所有模型的每次测试集）结果
    test_nfolds = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    #kf.split(x_train)返回一个包含n个元组的列表，每个元组是(train_index, test_index)其中train_index、test_index都是索引列表，enumerate会给n个元组加上索引
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        #测试集
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        #验证集
        x_tst, y_tst = x_train[test_index], y_train[test_index]

        clf.fit(x_tra, y_tra)
        #预测验证并填充second_train，循环之后所有的训练集数据都被验证过了
        second_train[test_index] = clf.predict(x_tst)
        #预测测试集x_test并用均值填充second_test
        test_nfolds[:, i] = clf.predict(x_test)
        second_test[:] = test_nfolds.mean(axis=1)
    #返回一个所有验证集的结果 和 所有模型测试集结果的均值
    return second_train, second_test

#导入模型
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
rf_model = RandomForestClassifier()
adb_model = AdaBoostClassifier()
gdbc_model = GradientBoostingClassifier()
et_model = ExtraTreesClassifier()
svc_model = SVC()

#造数据
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.2)

#存放所有模型的结果
train_sets = []
test_sets = []
for clf in [rf_model, adb_model, gdbc_model, et_model, svc_model]:
    train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
    train_sets.append(train_set)
    test_sets.append(test_set)

#把所有训练集结果按列合并
meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)

#使用决策树作为我们的次级分类器
from xgboost import XGBClassifier
xgb = XGBClassifier()

xgb.fit(meta_train, train_y)

y_score = xgb.predict(meta_train)
y_score_test = xgb.predict(meta_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(train_y,y_score))
print(accuracy_score(test_y,y_score_test))

# 上面是全量数据的融合
# 如果数据是训练集和预测集，且预测集没标签的形式，上面就是训练集（自己分训练集、验证集）的融合，下面也要对测试集进行同样的构造得到次级训练集
test_data = test_data.fillna(0)
sets = []
for clf in [rf_model, adb_model, gdbc_model, et_model, svc_model]:
    set = clf.predict(test_data.values)
    sets.append(set)
data = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in sets], axis=1)
```



##### GBDT+LR

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

gbdt= GradientBoostingClassifier()
gbdt.fit(df_x,df_y)

leaves= gbdt.apply(df_x)[:,:,0]
X_train,X_test,y_train,y_test = train_test_split(leaves,df_y,random_state=7,test_size=0.33)

lr= LogisticRegression()
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)
```



##### XGB+LR

```python
import xgboost as xgb
from sklearn.preprocessing import  OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

model = xgb.XGBClassifier(learning_rate=0.08,n_estimators=50,max_depth=5,gamma=0,subsample=0.9,colsample_bytree=0.5) 
model.fit(X_train.values, y_train.values)

X_train_leaves = model.apply(X_train).astype(np.int32)
X_test_leaves = model.apply(X_test).astype(np.int32)

X_tran_onthot = OneHotEncoder().fit_transform(X_leaves)
X_test_onehot = OneHotEncoder().fit_transform(X_test_leaves)

lr = LogisticRegression()
lr.fit(X_tran_onthot, y_train)

y_pred = lr.predict_proba(X_test_onehot)[:, 1]
auc = roc_auc_score(y_test, y_pred)
```



## 集成树模型

### 1、随机森林原理

二分类、多分类、回归



随机森林是由很多决策树（cart）构成的，不同决策树之间没有关联。



**一句话介绍**：

当我们进行分类任务时，新的输入样本进入，就让森林中的每一棵决策树分别进行判断和分类，每个决策树会得到一个自己的分类结果，决策树的分类结果中哪一个分类最多，那么随机森林就会把这个结果当做最终的结果。

Random Forest 是经典的基于 Bagging 框架的模型，并在此基础上引入特征采样和样本采样来降低基模型间的相关性，更加发挥了Bagging 降低方差的作用。



**自助采样法**：

样本容量为N的样本，有放回的抽取N次，共抽取N个样本。

对于一个样本，它在某一次含 N 个样本的随机采样中，每次被采集到的概率是 1/N。不被采集到的概率为 1−1/N。

若 N 次采样都没有被采集中的概率是1/N (1−1/N)，则当 N→∞时，1/N (1−1/N)的极限值约为0.368，即每轮采样中，训练集中大约有 36.8% 的数据没有被采中。



**软硬投票：**

软投票是概率的集成，硬投票是结果标签的集成。

软投票：如果算法 1 预测对象是一块岩石的概率是 40% ，算法 2 是80% ，那么集成模型将预测该对象具有 (80 + 40) / 2 = 60% 是岩石的可能性。

硬投票：如果三个算法将特定葡萄酒的颜色预测为“白色”、“白色”和“红色”，则集成将预测“白色”。



**步骤：**

1、**自助采样**法（Bootstrap）：样本容量为N的样本，有放回的抽取N次，共抽取N个样本，作为其中一个决策树的训练数据。

2、选取特征：选择全部或者随机的部分特征，按照决策树分裂的策略（如基尼系数）进行分裂。

3、形成决策树：循环第2步，形成决策树，注意决策树**不剪枝**。

4、投票：分类任务使用软投票，回归任务使用每颗树结果的均值。



**优点：**

1、因为自助采样和选取部分特征，还有多颗决策树结果投票，**不容易过拟合**。

2、易于并行化，在大数据集上有很大的优势。

3、可以处理很多特征的数据，可以减少特征选择的工作量。

**缺点**：高基数特征，在创建分裂点时，会导致分支数量和深度增加，然后导致过拟合，也使得高基数特征重要性更高。



### 1、RandomForest

随机森林属于集成模型中的bagging。

```python
# 二分类，多分类，排序
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X, Y)
clf.get_parms()

# 回归
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
```

```python
# 决策树分类器的集合
clf.estimators_

# 类别标签数组
clf.classes_
# 类别标签个数
clf.n_classes_

# 特征数量
clf.n_features_
clf.max_features_

# 输出结果数量
clf.n_outputs_

# 特征重要性
clf.feature_importances_

# 包外测试集得分
clf.oob_score_ 
```

```
n_estimators : 随机森林中树的个数，即学习器的个数。
max_depth：决策树的最大深度
max_features：随机抽取的候选划分属性集的最大特征数（属性采样）

criterion：选择最优划分属性的准则，默认是"gini"，可选"entropy"
min_samples_split：内部节点再划分所需最小样本数。默认是2，可设置为整数或浮点型小数。
min_samples_leaf：叶子节点最少样本数。默认是1，可设置为整数或浮点型小数。如果某叶子节点数目小于这个值，就会和兄弟节点一起被剪枝。
max_leaf_nodes：最大叶子结点数。默认是不限制。
min_impurity_split：节点划分最小不纯度。是结束树增长的一个阈值，如果不纯度超过这个阈值，那么该节点就会继续划分，否则不划分，成为一个叶子节点。
min_impurity_decrease : 最小不纯度减少的阈值，如果对该节点进行划分，使得不纯度的减少大于等于这个值，那么该节点就会划分，否则，不划分。

random_state ：随机种子数。
bootstrap :自助采样法，有放回的采样，大量采样的结果就是初始样本的63.2%作为训练集。默认选择自助采样法。
oob_score:是否采用袋外样本来评估模型的好坏，默认值False,袋外样本误差是测试数据集（即bootstrap采样剩下的36.8%的样本）误差的无偏估计，所以推荐设置True。
```



### 2、AdaBoost原理

AdaBoost（Adaptive Boosting，自适应增强），其自适应在于：前一个基本分类器分错的样本会得到加强，加权后的全体样本再次被用来训练下一个基本分类器。同时，在每一轮中加入一个新的弱分类器，直到达到某个预定的足够小的错误率或达到预先指定的最大迭代次数。



##### 算法流程

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/v2-5bc052f6288e0efbf0038e3e8e86bafc_r.jpg)

步骤如下：初始化数据权重 -->  训练第一个分类器，结合权重计算最小错误率 --> 计算所有模型总最小错误率，计算正确数据与错误数据的比例 --> 利用计算正确数据与错误数据的比例更新数据权重 --> 循环。

预测：训练更新了每条数据的权重w，更新过程中利用w计算出每个分类器的权重a，分类器权重和模型概率相乘后，都进行相加，即得到总的分类概率。



假如我们有100条数据，有3个分类器，即M=3。预测两个label，我们设为1和-1。具体步骤如下：

【1】初始化权重w ， w<sub>n</sub>代表第n条数据的权重。一开始所有数据权重都是1/N，其中N是数据条数。给每个数据的权重都是0.01。

【2】M为classifiers的个数。开始当前训练器，假设m=1，方便理解。

【3-4】让classifier适应数据，使得预测出来的结果错误率最少，即准确率最高。例如classifier A，通过调整了三次参数，分别预测正确了70个，75个，68个，则使得 J<sub>m</sub> 最小的是第二次，错误个数为25，此时  J<sub>m</sub>=(25∗0.01)=0.25。

【5】计算总体误差，计算方法是把错的数据的权重加起来，除以所有正确和错误数据总权重的和。这时 ε<sub>m</sub>=(25∗0.01)/(100∗0.01) 。虽然这里看起来和上一步结果一样，这是因为刚开始权重一样，在第二个classifier后就会不一样了。

【6-7】 定义损失函数为指数损失
$$
L(y,F) = \sum_\limits{i=1}^{n}exp(-y_iF_{k}(x_i)) \\
$$
根据损失函数最小，分类器权重a 计算为 正确数据与错误数据的比例。
$$
a_m = log(\frac{1-\epsilon}{\epsilon})
$$

$$
a_m = log(\frac{1-0.25}{0.25}) = log(3)=1.098
$$



 然后对于归类错误的，他们的新权重为
$$
w_n^{(m+1)} = exp(log(3)) *w_n^{(m)}= 3*w_n^{(m)}
$$
【8】训练下一个classifier C2, C3

【9】预测过程如下。比如classifier C1预测结果 y1=0.8，也就是80%这个预测值是1，此时计算产生的 a1=log(3) = 1.098。然后 C2（包括C1）结果是预测 a2=1.2,预测值y2=-0.1,然后 a3=1.5,预测值为0.1,则最后输出 
$$
Y_m=Sign(1.098*0.8+1.2*(-0.1)+1.5*(0.1)) = Sign(0.9084) = 1
$$
预测值为1。



##### 名词解释

加法模型：最终的强分类器是由若干个弱分类器加权平均得到的。

前向分布学习算法：算法是通过一轮轮的弱学习器学习，利用前一个弱学习器的结果来更新后一个弱学习器的训练集权重。第 k 轮的强学习器为
$$
F_{k}(x)=\sum_{i=1}^{k}\alpha_i f_i(x)=F_{k-1}(x)+\alpha_{k}f_k(x) \\
$$


##### 优缺点

1. 分类精度高；
2. 可以用各种回归分类模型来构建弱学习器，非常灵活；
3. 不容易发生过拟合。



1. 对异常点敏感，异常点会获得较高权重。



### 2、AdaBoost

```python
# 分类
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100)

# 回归
from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor(n_estimators=50)
```



### 3、GBDT原理

GBDT（Gradient Boosting Decision Tree）是多颗树组成的boosting算法。GBDT中的树是cart回归树，GBDT用来做回归预测，调整后也可以用于分类。

GBDT的思想使其具有天然优势可以发现多种有区分性的特征以及特征组合。



复习一下cart进行回归的算法

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/967544-b768a350d5383ccb.png)

GBDT

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/967544-37a15b71dc6f6ca3.png)



该算法的核心思想是：在每一轮迭代中，利用损失函数相对于当前模型预测值的负梯度作为“伪残差”（pseudo-residual），拟合一个回归树来近似这个残差（真实值和预测值之间的差值），并通过加和每棵树的贡献不断提升模型性能。



第一步，初始化第一颗树的参数，γ是预测值，L 是损失函数，可能是平方差损失或者交叉熵损失。

第二步，m代表第m颗树

​	a，求得伪残差作为第m颗树的y。计算上一颗树 损失函数 关于 预测值 的负梯度 作为伪残差。

​			*为什么使用伪残差，借鉴了梯度下降的思想，伪残差可以使得损失函数最快减小。*

​	b，生成第m颗cart树。

​	c，计算第m颗树每个叶子节点的值。即求  真实值 和上一颗树的输出+这棵树的叶子节点值 的损失函数 最小时。

​	d，更新树。

第三步，输出模型



交叉熵损失的GBDT

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/v2-2bba06b6b3be74b1ee5aca4ce5ee7273_r.jpg)



### 3、GBDT

```python
# 分类
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0).fit(X_train, y_train)

# 回归
from sklearn.ensemble import GradientBoostingRegressor
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1,loss='ls').fit(X_train, y_train)
```



| 参数              | 默认值 | 解释                                                         |
| ----------------- | ------ | ------------------------------------------------------------ |
| learning_rate     | 0.1    | 每棵树更新时的步长，控制每次迭代对最终模型的贡献。较低值防止过拟合，但需更多树。 |
| n_estimators      | 100    | 弱学习器（树）的总数，迭代次数，通常与learning_rate成反比。  |
| max_depth         | 3      | 单棵树的最大深度，用于控制树的复杂度，防止过拟合。           |
| subsample         | 1.0    | 每棵树训练时随机采样的样本比例，低于1.0可增加随机性，提高泛化能力。 |
| colsample_bytree  | 1.0    | 每棵树构建时随机采样的特征比例，降低特征间相关性，提升模型稳健性。 |
| min_samples_split | 2      | 内部节点分裂所需的最小样本数，用于限制过度分裂。             |
| min_samples_leaf  | 1      | 叶节点中最少的样本数，确保叶节点有足够样本以避免噪声影响。   |
| loss              | 'ls'   | 损失函数，'ls'代表最小二乘损失（均方误差），回归任务中常用；分类中通常用'deviance'。 |



### 3、XGB面试题



XGBoost 与传统 GBDT 有哪些不同？

- **基分类器**：GBDT 通常仅使用 CART，而 XGBoost 除了 CART 还支持线性分类器；
- **导数信息**：GBDT 只用一阶梯度，而 XGBoost 采用二阶泰勒展开（利用一阶和二阶信息），使得损失函数近似更精准；
- **正则项**：XGBoost 在目标函数中显式加入正则化项，有效防止过拟合；
- **缺失值处理与列采样**：XGBoost 自动学习缺失值的分裂方向，并支持列采样，有助于降低噪声和过拟合；
- **并行化**：XGBoost 在特征维度上实现并行查找最佳分割点，加快训练速度。



为什么 XGBoost 采用二阶泰勒展开？

二阶展开的优势在于：

- **精度更高**：相比只使用一阶导数，二阶信息能更准确地近似真实损失；
- **可扩展性**：支持自定义损失函数（只要求一阶、二阶可导）；
- **加快收敛**：利用二阶信息类似于拟牛顿法，能更快找到下降方向。



XGBoost 如何处理缺失值？

在树构建过程中，XGBoost 对每个分裂节点会同时尝试将缺失值分到左右两侧，并选取能获得最大增益的方向作为默认分裂方向，从而自动适应缺失数据。



XGBoost 为什么能实现并行训练？

虽然整个 boosting 过程是串行的，但 XGBoost 在构建单棵树时对每个结点的最佳分裂点搜索采用了特征维度的并行计算。预先对各特征进行排序并存储为 block 结构，使得在查找最佳分裂点时，可以多线程同时计算各特征的分裂增益，从而大幅提升效率。


### 3、XGBooost

官网：https://xgboost.readthedocs.io/en/latest/

调参：https://zhuanlan.zhihu.com/p/33384959

```python
import xgboost

clf = xgboost.XGBClassifier()

# 模型训练
clf.fit(X_train,y_train)

# 模型预测
y_prob = clf.predict_proba(X_test)[:,1] 
y_pred = clf.predict(X_test) 

# 获取特征重要性图
from xgboost import plot_importance
plot_importance(clf,max_num_features=5)

# 获取特征重要性表，并排序
df_importance = pd.DataFrame()          
df_importance['column_name'] = train_x.columns
df_importance['importance'] = clf.feature_importances_
df_importance.sort_values(by='importance')
```



XGBoost 参数调优的一般步骤

1. **调整学习率与树的数量**：先设定一个较大的树数和较小学习率；
2. **调节 max_depth 与 min_child_weight**：控制树的复杂度；
3. **gamma 参数**：设定最小分裂损失，筛选有效分裂；
4. **子采样与列采样**：设置 subsample 与 colsample_bytree 降低过拟合风险；
5. **正则化参数**：调节 L1（alpha）和 L2（lambda）正则化力度；
6. **综合验证**：结合交叉验证调整各参数。



sklearn--XGBClassifier 参数

```python
#booster 基分类器
    gbtree 树模型做为基分类器（默认）
    gbliner 线性模型做为基分类器
#nthread=1  所用cpu核数
    nthread=-1时，使用全部CPU进行并行运算（默认）
#scale_pos_weight	正样本的权重，
	在二分类任务中，当正负样本比例失衡时。scale_pos_weight=负样本数量/正样本数量。

#n_estimatores=100
	基模型个数 
#early_stopping_rounds=5
	当迭代5次后，仍没有提高，提前停止训练
#max_depth=6
	树的深度，默认值为6，典型值3-10。
#learning_rate=0.3
	学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
#alpha
    L1正则化系数，默认为1
#lambda
    L2正则化系数，默认为1
 
#min_child_weight=1
	值越小，越容易过拟合
#subsample=1  抽样本比例
	训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。防止overfitting。
#colsample_bytree  抽特征比例
	训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。防止overfitting。
#gamma	惩罚项系数
	指定节点分裂所需的最小损失函数下降值。默认0

#objective 目标函数
    回归任务
        reg:linear (默认)
        reg:logistic 
    二分类
        binary:logistic     概率 
        binary：logitraw   类别
    多分类
        multi：softmax  num_class=n   返回类别
        multi：softprob   num_class=n  返回概率
    rank:pairwise 
#eval_metric
    回归任务(默认rmse)
        rmse--均方根误差
        mae--平均绝对误差
    分类任务(默认error)
        auc--roc曲线下面积
        error--错误率（二分类）
        merror--错误率（多分类）
        logloss--负对数似然函数（二分类）
        mlogloss--负对数似然函数（多分类）
```



## ---- ---- ---- ----



## 多输出模型

##### 概述

regression（回归）

Binary classification（二分类）

Multiclass classification（多分类）

Multioutput regression （多输出回归）

Multilabel classification（多标签分类/多输出二分类）

Multioutput-multiclass classification（多输出多分类）/ multi-task classification（多任务分类）



##### 二分类、多分类

sklearn 的所有分类器对这些输出开箱即用

```
固有多分类:
        sklearn.naive_bayes.BernoulliNB
        sklearn.tree.DecisionTreeClassifier
        sklearn.tree.ExtraTreeClassifier
        sklearn.ensemble.ExtraTreesClassifier
        sklearn.naive_bayes.GaussianNB
        sklearn.neighbors.KNeighborsClassifier
        sklearn.semi_supervised.LabelPropagation
        sklearn.semi_supervised.LabelSpreading
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis
        sklearn.svm.LinearSVC (setting multi_class=”crammer_singer”)
        sklearn.linear_model.LogisticRegression (setting multi_class=”multinomial”)
        sklearn.linear_model.LogisticRegressionCV (setting multi_class=”multinomial”)
        sklearn.neural_network.MLPClassifier
        sklearn.neighbors.NearestCentroid
        sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
        sklearn.neighbors.RadiusNeighborsClassifier
        sklearn.ensemble.RandomForestClassifier
        sklearn.linear_model.RidgeClassifier
        sklearn.linear_model.RidgeClassifierCV
1对1-多分类:
        sklearn.svm.NuSVC
        sklearn.svm.SVC.
        sklearn.gaussian_process.GaussianProcessClassifier (setting multi_class = “one_vs_one”)
1对多-多分类:
        sklearn.ensemble.GradientBoostingClassifier
        sklearn.gaussian_process.GaussianProcessClassifier (setting multi_class = “one_vs_rest”)
        sklearn.svm.LinearSVC (setting multi_class=”ovr”)
        sklearn.linear_model.LogisticRegression (setting multi_class=”ovr”)
        sklearn.linear_model.LogisticRegressionCV (setting multi_class=”ovr”)
        sklearn.linear_model.SGDClassifier
        sklearn.linear_model.Perceptron
        sklearn.linear_model.PassiveAggressiveClassifier
```



##### 多输出二分类

y有多个标签，如新闻主题可以为 政治、金融、 教育 中的一个或几个。

```python
#将多标签分类转化为多输出二分类的格式
from sklearn.preprocessing import MultiLabelBinarizer
y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]

MultiLabelBinarizer().fit_transform(y)

>>> array([[0, 0, 1, 1, 1],
       [0, 0, 1, 0, 0],
       [1, 1, 0, 1, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 0, 0]])
```

```python
#多标签分类/多输出二分类
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

X, Y = make_multilabel_classification(n_samples=10, n_features=5, n_classes=3, n_labels=2)
X_train, X_test, Y_train ,Y_test = train_test_split(X, Y, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
cls = DecisionTreeClassifier()
cls.fit(X_train, Y_train)

Y_pred = cls.predict(X_test)
```

```
多标签分类/固有的多输出二分类:
        sklearn.tree.DecisionTreeClassifier
        sklearn.tree.ExtraTreeClassifier
        sklearn.ensemble.ExtraTreesClassifier
        sklearn.neighbors.KNeighborsClassifier
        sklearn.neural_network.MLPClassifier
        sklearn.neighbors.RadiusNeighborsClassifier
        sklearn.ensemble.RandomForestClassifier
        sklearn.linear_model.RidgeClassifierCV
```



##### 多输出回归

1.固有多输出回归

```
sklearn.linear_model.LinearRegression
sklearn.neighbors.KNeighborsRegressor
sklearn.tree.DecisionTreeRegressor
sklearn.ensemble.RandomForestRegressor
```

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)
data_in = [[-2.0222, 0.3156, 0.82797464, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]]

model = LinearRegression()
model.fit(X, y)

yhat = model.predict(data_in)
```



2.每一输出单独建立一个回归模型

MultiOutputRegressor

```python
#假设每个输出之间都是相互独立的
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_regression(n_samples=10, n_targets=3, random_state=1)

MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X, y).predict(X)

>>> array([[-154.75474165, -147.03498585,  -50.03812219],
       [   7.12165031,    5.12914884,  -81.46081961],
       [-187.8948621 , -100.44373091,   13.88978285],
       [-141.62745778,   95.02891072, -191.48204257],
       [  97.03260883,  165.34867495,  139.52003279],
       [ 123.92529176,   21.25719016,   -7.84253   ],
       [-122.25193977,  -85.16443186, -107.12274212],
       [ -30.170388  ,  -94.80956739,   12.16979946],
       [ 140.72667194,  176.50941682,  -17.50447799],
       [ 149.37967282,  -81.15699552,   -5.72850319]])
```



3.每个输出都参与的链接模型

RegressorChain

```python
#第一个模型的输入和输出作为第二个模型的输入，以此类推
from sklearn.datasets import make_regression
from sklearn.multioutput import RegressorChain
from sklearn.svm import LinearSVR

X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)
data_in = [[-2.02220, 0.3156, 0.8279, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]]

model = LinearSVR()
wrapper = RegressorChain(model)

wrapper.fit(X, y)

yhat = wrapper.predict(data_in)
```





##### 多输出多分类

1.固有的多输出多分类

```
sklearn.tree.DecisionTreeClassifier
sklearn.tree.ExtraTreeClassifier
sklearn.ensemble.ExtraTreesClassifier
sklearn.neighbors.KNeighborsClassifier
sklearn.neighbors.RadiusNeighborsClassifier
sklearn.ensemble.RandomForestClassifier
```



2.每一个输出单独建立一个分类模型

```python
#假设每个输出之间都是相互独立的
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np

X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
y2 = shuffle(y1, random_state=1)
y3 = shuffle(y1, random_state=2)
Y = np.vstack((y1, y2, y3)).T

forest = RandomForestClassifier(n_estimators=100, random_state=1)

multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)

multi_target_forest.fit(X, Y).predict(X)
```



3.ClassifierChain

见多输出回归对应部分或者sklearn官网。



## 过拟合

### 什么是过拟合

![image-20220802174103142](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220802174103142.png)

欠拟合，模型偏差很高，在训练集上和预测集上表现都不好且相近。

泛化能力强，模型在训练集合测试集上都变现良好。

过拟合，在训练集上表现良好，在测试集上表现不好，方差很高。



偏差代表预测结果的准确性，方差代表预测结果的波动性。



过拟合和欠拟合也可以从模型容量和数据复杂度之间的关系，欠拟合就是简单模型不能拟合复杂数据。



### 解决过/欠拟合

过拟合：使用 增加数据，简化模型 解决

1. 获得更多的训练样本——解决高方差（过拟合）
2. 尝试减少特征的数量——解决高方差（过拟合）
3. 尝试增加正则化程度λ——解决高方差（过拟合）
4. 超参数设置的使模型更简单 （DropOut、Early stopping ）
5. 多次划分数据，训练



欠拟合：使用 复杂模型 解决

1. 尝试获得更多的特征——解决高偏差（欠拟合）
2. 尝试减少正则化程度λ——解决高偏差（欠拟合）
3. 尝试增加多项式特征——解决高偏差（欠拟合）
4. 超参数设置的使模型更复杂



## 正则化



### 什么是正则化

过拟合发生后，需要惩罚一些高次项或者不合理特征的参数的值。

因为不知道具体惩罚哪一项的参数，所以对所有项的参数进行惩罚，让代价函数最优化的算法计算惩罚的程度。

![image-20220811112714679](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220811112714679.png)

目标函数 = 代价函数+正则化项

λ越大，参数w就得越小，才能使得代价函数最小。

这样，就保证了求出来参数w小，避免了过拟合的风向。



### 梯度下降

正则化之后使用梯度下降求目标函数最小值

![image-20220815110534721](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20220815110534721.png)



### L1&L2正则

L1正则是在损失函数中  添加参数的绝对值之和的惩罚项		λ * Σ|w_i|

L2正则是在损失函数中  添加参数平方和的惩罚项					λ * Σ(w_i^2)



L1正则与L2正则的区别：

1. L1正则导致稀疏解，有助于特征选择；L2正则不会将权重压缩至0，但有助于防止过拟合。
2. L1正则的损失函数不可导（在0点处），而L2正则的损失函数可导。



如果需要进行特征选择，即筛选出重要特征，可以考虑使用L1正则。如果不需要特征选择，而只关心防止过拟合，可以考虑使用L2正则



# 半监督学习

`LabelPropagation`和`LabelSpreading`可以预测y缺失的数据集，只要把缺失部分换成-1即可预测

```python
#以下代码用于比较LabelSpreading预测y缺失数据集的准确度
#点的颜色是真实值。背景色是预测值
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.semi_supervised import LabelSpreading

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
print(y)

#准备缺失y
rng = np.random.RandomState(0)
y_30 = np.copy(y)
y_30[rng.rand(len(y)) < 0.3] = -1
y_50 = np.copy(y)
y_50[rng.rand(len(y)) < 0.5] = -1

#准备模型
ls30 = (LabelSpreading().fit(X, y_30), y_30)
ls50 = (LabelSpreading().fit(X, y_50), y_50)
ls100 = (LabelSpreading().fit(X, y), y)
rbf_svc = (svm.SVC(kernel='rbf', gamma=.5).fit(X, y), y)

#准备画图
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
titles = ['Label Spreading 30% data',
          'Label Spreading 50% data',
          'Label Spreading 100% data',
          'SVC with rbf kernel']
color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}


for i, (clf, y_train) in enumerate((ls30, ls50, ls100, rbf_svc)):
    plt.subplot(2, 2, i + 1)

    #根据预测结果绘制背景
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    #每个点的x，y轴是特征，颜色是真实y
    colors = [color_map[y] for y in y_train]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black')
    plt.title(titles[i])

plt.suptitle("Unlabeled points are colored white", y=0.1)
plt.show()
```



# 无监督学习



## 聚类模型

聚类一定要选好的特征进行聚类，才能使结果具有可解释型。



### 聚类模型选择



**K-means聚类：**

1、需要预先指定簇的数量。

2、 计算复杂度较低，适用于大规模数据集。

3、 对初始中心点敏感，对异常值敏感。

4、适用于数据点呈球状分布、簇大小相近、密度相似的数据集。

5、通过最小化簇内平方误差（即样本点到簇中心的距离平方和）来划分数据。



**层次聚类：**

1、无需预先指定簇的数量。

2、计算复杂度较高，不适用于大规模数据集。

3、对噪声敏感，对异常值敏感。

4、适用于 任意形状簇 的数据集。

5、生成嵌套的聚类树状图，揭示数据的多层次关系。



**DBSCAN聚类：**

1、无需预先指定簇的数量。

2、在处理高维数据时效率较低，但在低维数据中表现良好。

3、对噪声不敏感，对异常值不敏感。

4、适用于 任意形状簇、簇密度相似 的数据集。

5、基于密度进行聚类，通过检测高密度区域来形成簇。



### K-means 原理



#### 聚类步骤

1、首先选择n个随机的点，称为聚类中心（cluster centroids）。

2、对于数据集中的每一个数据，按照距离个中心点的距离，将其与距离最近的中心点关联起来，与同一个中心点关联的所有点聚成一类。

3、计算每一个组的平均值，将该组所关联的中心点移动到平均值的位置。

4、重复步骤2-3直至中心点不再变化。



#### 代价函数

代价函数：所有的数据点与其所关联的聚类中心点之间的距离之和。

![image-20230111135505042](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20230111135505042.png)



#### 随机初始化

1. k<m，即聚类中心点的个数要小于所有训练集实例的数量
2. 初始k个聚类中心，为随机选择的k个训练实例



K-均值的一个问题在于，它有可能会停留在一个局部最小值处，而这取决于初始化的情况。

我们通常需要多次运行K-均值算法【k较小的时候（2--10）】，每一次都重新进行随机初始化，最后再比较多次运行K-均值的结果，选择代价函数最小的结果。



#### 选择聚类数

1、根据业务目的，人工确定聚类中心数量。

2、肘部选择法，画图，横轴是聚类中心数量，纵轴是代价函数，寻找曲线拐点就是最佳的聚类数量。



#### 优缺点

1. 需要预先指定聚类的簇数，如果簇的数量无法确定，那么需要采用一些启发式方法进行确定；
2. 在大数据集上，算法的速度很快；支持进行分布式计算；
3. 对于异常点比较敏感，需要在算法之前对异常点进行处理；
4. 初始聚类中心的选择对最终结果影响很大，不同的初始化方式可能会得到不同的结果。
5. 适用于数据点呈球状分布、簇大小相近、密度相似的数据集。



### K-means 代码

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=1500, random_state=42)
y_pred = KMeans(n_clusters=2, random_state=42).fit_predict(X)
```

```python
# 查看聚类是否均匀
df.groupby('y_pred').count()

# 查看聚类中心，根据每一个属性不同类的差异进行解释
pd.DataFrame(model.cluster_centers_ )

# 去除每一类中的重复值查看
df.loc[df['y_pred']==5,['institutes_type','big_area','year_price','avg_date']].drop_duplicates()
```



### 肘部观察法

画图，横坐标为聚类数，纵坐标为代价函数。拐点处，即为参考的聚类数量。

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/f3ddc6d751cab7aba7a6f8f44794e975.png)



使用肘部观察法选择最佳的聚类群数

```python
from yellowbrick.cluster.elbow import kelbow_visualizer

kelbow_visualizer(KMeans(random_state=42),df,k=(2,15))
```



### 层次聚类原理



#### 聚类步骤

1. **初始化**：将每个数据点视为一个独立的簇，因此对于包含 n 个数据点的数据集，初始时有 n 个簇。 

   

2. **计算距离矩阵**：根据选定的距离度量（如欧几里得距离），计算并构建所有簇之间的距离矩阵。

   

3. **合并最近的簇**：在距离矩阵中，找到距离最小的两个簇，将它们合并为一个新的簇。 

   

4. **更新距离矩阵**：根据所选的链接方法（如单链接、完全链接、平均链接等），更新 新簇与其他簇 之间的距离。

   

5. 重复执行步骤3和4，直到所有数据点都被合并到一个簇中，或达到预定的簇数。

   

层次聚类的结果通常以树状图（dendrogram）的形式表示，展示了数据点的聚类过程和层次结构，可通过观察树状图，确定最佳的聚类数量。



#### 合并策略

最小链接法（Single Linkage）：定义两个簇之间的距离为 各自成员间最短的距离，即最近样本点之间的距离

优点是速度快。缺点是倾向于形成链状的簇结构，链状的簇结构可能并不是好的聚类。对噪声和离群点较为敏感。 



最大链接法（Complete Linkage）：定义两个簇之间的距离为 各自成员间最长的距离，即最远样本点之间的距离。

通常生成紧凑且球形的簇。适用于分布不规则的数据集，对类别分布均匀的数据集效果差。不会和离群点合并，但容易把噪声数据当作单独簇，也需预处理。



平均链接法（Average Linkage）：定义两个簇之间的距离为 两个簇中所有数据点之间的距离的平均值。

适用于分布不规则的数据集，不容易受到噪声数据的影响。缺点是计算复杂度较高。



中心链接法（Centroid Linkage）：定义两个簇之间的距离为 两个簇的质心（簇中样本坐标的均值）之间的距离。

适用于分布不规则的数据集，不容易受到噪声数据的影响。缺点是计算复杂度较高，且对簇的形状和密度要求较高。



矩阵链接法（Ward Linkage）：定义两个簇之间的距离为 簇与簇合并后所有数据的方差。

适用于分布不规则的数据集，不容易受到噪声数据的影响。适用于发现大小相似的类簇，适用于最小化簇内方差的场景。 



#### 最佳聚类数

1. 观察树状图：在树状图中，横轴是数据点，垂直轴表示簇之间的合并距离。较长的垂直线段表示在较大的距离上合并的簇，在较长的垂直线段处进行切割，可以得到具有较高相似性的类簇。依赖主观判断。

   ![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/v2-c2e81c22f5b0b3a314799ec645efd2ac_1440w.jpg)

   

2. 使用内部评估指标：可以通过计算不同聚类数量下的评估指标值来选择最佳的聚类数量。轮廓系数值越接近1，Davies-Bouldin 值越小，Calinski-Harabasz 值越大，聚类效果越好。

   

3. 使用“肘部法则”（Elbow method）：绘制簇数量与簇内部误差平方和的关系图，寻找图形中的“肘部”点，该点通常表示最佳聚类数量。



### 层次聚类代码

```python
from sklearn.cluster import AgglomerativeClustering

#single,complete,average,Ward	四种策略
clustering = AgglomerativeClustering(linkage='ward', n_clusters=2).fit(X)

clustering.labels_
```

通过观察聚类树，确定聚类数量

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 计算簇的关系
Z = linkage(X, method='ward')

# 绘制聚类树，设置层次聚类的参数
plt.figure(figsize=(10, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

# 绘制树状图
dendrogram(Z, leaf_rotation=90,leaf_font_size=8)
plt.show()
```



### DBSCAN 原理

DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是一种基于密度的聚类算法。



#### 聚类步骤

1. 初始化参数：DBSCAN算法有两个关键参数，即半径ε（Epsilon）和最小点数MinPts。半径ε定义了一个点的邻域范围，而最小点数MinPts表示一个点的邻域内至少要有多少个点才能认为它是一个核心点。

   

2. 计算邻域：对于数据集中的每个点，计算其ε邻域，即与该点距离小于或等于ε的所有点的集合。

   

3. 寻找核心点：根据MinPts参数，找出所有的核心点。一个核心点的邻域内至少包含MinPts个点（包括核心点本身）。

   

4. 创建类簇：从一个未被访问的核心点开始，将其标记为已访问。然后将该核心点的所有密度可达点（即邻域内的点）归为同一个类簇。对于这些密度可达点，如果它们也是核心点，则将它们的密度可达点也归为该类簇。重复这个过程，直到所有密度可达点都被归为该类簇。

   

5. 处理噪声点：当所有的核心点都被访问并分配到相应的类簇后，剩余的未被分配的点被认为是噪声点。可以将它们归为一个特殊的噪声类簇，或者直接将它们排除在聚类结果之外。

   

6. 返回结果：返回所有找到的类簇以及噪声点（如果有）。此时，聚类过程完成。



#### 初始化参数

初始化参数：半径ε、最小点数MinPts



1. 确定MinPts：MinPts的选择与数据集的维数和噪声水平有关。一般来说，较大的MinPts值会导致较少的类簇和较高的噪声鲁棒性。

   ​						一个常用的经验法则是将MinPts设置为数据集维数的2倍加1（即2 * d + 1），至少为 d+1，但这个值可能不适用于所有情况。

   

2. k-距离图 确定 半径ε：

   ​		1）计算每个数据点到其第 k 近邻的距离（这里的 k 通常取 MinPts 值。例如，若 MinPts = 4，则计算每个点到其第 4 近邻的距离），作为纵坐标。

   ​		2）对这些距离进行排序：将所有点的第 k 近邻距离按升序排列，作为横坐标。

   ​		3）绘制 k-距离图：在图中，横轴表示数据点（按排序顺序），纵轴表示对应的第 k 近邻距离。

   ​		4）在绘制的 k-距离图中，寻找曲线的“拐点”（elbow point），即曲线陡然上升的位置。该拐点对应的距离值即为 ε 的最佳估计值

   

3. 使用领域知识：可以利用领域知识来选择合适的ε和MinPts值。

   

4. 试验与调整：尝试不同的ε和MinPts组合，使用评估指标（如轮廓系数、Davies-Bouldin指数等），根据效果来确定参数。



### DBSCAN 代码

```python
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

y_pred = DBSCAN(eps = 0.1, min_samples = 10).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
```



### 距离计算

相似度/距离计算方法



#### 欧几里得距离

Euclidean Distance：m维空间中两个点之间的直线距离

![image-20230111143439015](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20230111143439015.png)



#### 标准化欧氏距离 

Standardized Euclidean Distance：先将各个分量都“标准化”到均值、方差相等，即使得各个维度分别满足标准正态分布，再进行欧式距离计算。

![image-20230130173152630](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20230130173152630.png)

其中，x是修正过的x，s是标准差。

![image-20230130174924032](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20230130174924032.png)



#### 余弦距离

余弦相似度

![image-20230131103259036](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20230131103259036.png)

余弦距离

![image-20230131103341225](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20230131103341225.png)



#### 曼哈顿距离

Manhattan Distance：两点在南北方向上的距离加上在东西方向上的距离。

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/1676906-a7b980267fdd791c.png)



#### 杰卡德距离

Jaccard Distance：1- 杰卡德相似系数

杰卡德相似系数 J(A,B) ：两个集合A和B交集元素的个数在A、B并集中所占的比例。jaccard值越大说明相似度越高。

![clip_image013](https://raw.githubusercontent.com/zhanghongyang42/images/main/28144621-9d519167c5b54643ba662ea8d10e3d33.gif)



#### 汉明距离

汉明距离(Hamming Distance)：对两个相同长度的字符串（一般为数字）进行异或运算，并统计结果为1的个数，那么这个数就是汉明距离。



#### 切比雪夫距离

 Chebyshev Distance：是向量空间中的一种度量，二个点之间的距离定义为其各坐标数值差绝对值的最大值

![image-20230112092603618](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20230112092603618.png)



#### 闵氏距离

闵可夫斯基距离（Minkowski distance)：闵氏距离是对多个距离度量公式的概括性的表述。

![image-20230112104457184](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20230112104457184.png)

当p=1时， 就是曼哈顿距离。当p=2时， 就是欧氏距离。当p→∞时， 就是切比雪夫距离。

根据p的不同， 闵氏距离可以表示某一类/种的距离。



#### 马氏距离

https://zhuanlan.zhihu.com/p/46626607

Mahalanobis distance：又称为数据的协方差距离，是欧氏距离的一种标准化修正。与标准化欧氏距离不同的是它认为各个维度之间不是独立分布的。

![image-20230130175801522](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20230130175801522.png)

其中Σ 是X与Y的协方差矩阵，u是均值。

如果Σ 是单位矩阵，则马氏距离退化成欧式距离；
如果Σ 是对角矩阵，则称为归一化后的欧式距离。



## 降维模型

降维算法是指将高维数据映射到低维空间的技术，旨在在降低数据复杂性的同时尽可能保留原始数据的有用信息。



为什么要进行降维：

一、数据压缩：减少存储空间，加快算法计算。降低相似特征的冗余。

二、数据可视化：降低维数有助于进行数据可视化，但是降维后的意义需要自己挖掘。



常见的降维算法包括 ：主成分分析（PCA）、线性判别分析（LDA）



### PCA原理

https://www.bilibili.com/video/BV14a4y1a7Gd/?spm_id_from=333.337.search-card.all.click&vd_source=95e4d2371451c9804d5d5d30293ea8c6



#### 前置知识 - 协方差

**协方差公式**，两个特征可以计算协方差。
$$
\text{Cov}(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})
$$
协方差可以表示相关性。当协方差为正，XY正相关。当协方差为负，XY负相关。当协方差为0的时候，表示二者无线性关系。

*可以画两条正相关的线，来模拟计算协方差，得到上面的结论。*



多组变量时，可以用**协方差矩阵**表示：
$$
\Sigma =
\begin{bmatrix}
\text{Var}(X_1) & \text{Cov}(X_1, X_2) & \cdots & \text{Cov}(X_1, X_n) \\
\text{Cov}(X_2, X_1) & \text{Var}(X_2) & \cdots & \text{Cov}(X_2, X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_n, X_1) & \text{Cov}(X_n, X_2) & \cdots & \text{Var}(X_n)
\end{bmatrix}
$$


#### 前置知识 - 基变换

单位向量（方向向量）：经过原点的，模为1的向量。用来表示方向。如二维坐标系中（0,1）就是一个单位向量。

基：一组线性无关的向量集合。向量一般为单位向量，不同的基构成不同的坐标系。



向量A在B上的投影长度：lAlcos(a)

假设B向量为单位向量时：lBl=1，$BB^T = 1$

故向量A在B上的投影长度改写为：lAlcos(a) = lAllBlcos(a) = A·B



基变换：根据公式 lAlcos(a) = lAllBlcos(a) = A·B，可以把向量A 的坐标变换到 新基 向量B 的坐标系中。



示例：

![image-20250331165148261](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20250331165148261.png)

![image-20250331165004267](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20250331165004267.png)

m为样本数，n为特征数，k为新基数量，z是新的坐标，w是特征矩阵，也就是新的坐标系的矩阵。



#### 推导过程

**找坐标系问题** 转化为 **求方差最大问题** 转化为 **求协方差矩阵的特征值和特征向量**。

------

降维思路就是找到一个新的坐标系，使得 数据点在这个坐标系的每个坐标轴上，都保留最大的信息。

坐标轴要正交，保证每个坐标轴保留的信息不重复。

------

引入专业知识 基变换，来表述坐标系的变换：

PCA降维，就是通过找到一个特征矩阵（坐标系变换的矩阵），使得变换后，新基在尽可能多的数据点周围，即所有数据在新基上每个单位向量上的投影坐标的 方差最大。

*特征矩阵的的向量要正交，特征矩阵的向量是单位向量。*

------

求方差最大过程：

1.数据去中心化(将数据平移到坐标轴中心，数据整体相对位置不变)，方便后续计算。

2.投影方差:
		m为样本数，n为原特征个数，k为需要降维的特征个数。

​		$x^i$= $[x^i_1,x^i_2]^T$  为第i组数据，用列向量表示。$x^i_j$表示第i组样本的第j个特征的值。

​		$u$= $[^i_1,u^i_2]^T$  为投影向量，并且$uu^T = 1$

​		$d_i = |x^i||u|cos = (x^i)^Tu$    为向量$x^i$在向量u的投影




![image-20250331174722155](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20250331174722155.png)

通过上面的公式变换，将求最大方差 转化为了 求协方差矩阵的形式

通过拉格朗日乘除法，将求最大方差 转化为了 求协方差矩阵的特征值和特征向量

![image-20250331180315097](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20250331180315097.png)



#### 计算步骤



1. **数据标准化**：将数据集中的每个特征进行标准化处理，使其均值为0，方差为1，以消除不同特征之间的量纲差异。

   

2. **计算协方差矩阵**：对标准化后的数据计算协方差矩阵。

   

3. **计算特征值和特征向量**：对协方差矩阵进行特征值分解，得到特征值和对应的特征向量。

   ​										特征向量代表新的坐标轴方向，特征值表示数据在这些方向上的方差大小。

   

4. **选择主要特征向量**：将特征值从大到小排序，选择前k个最大的特征值对应的特征向量，作为新的特征空间的基。

   

5. **转换数据**：将原始数据投影到选定的k个特征向量构成的新特征空间中，得到降维后的数据表示。



#### 主成分数量

计算方差贡献率大于95%或者99%：前 k 个特征向量所对应的特征值 / 所有特征值的和



### PCA代码

```python
#保证数据进行了标准化处理

from sklearn.decomposition import PCA
PCA(n_components=2).fit_transform(iris.data)

# 确定主成分数量
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
threshold = 0.95  # 设定累积方差贡献率的阈值，例如95%
num_components = np.argmax(cumulative_explained_variance >= threshold) + 1
print(f"需要保留的主成分数量: {num_components}")
```



## 异常检测



# 增量学习

模型在小批量数据上进行训练，适用于无法一次性加载到内存的大型数据集

```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB

# 假设我们有一个数据生成器，每次产生一批数据
def data_generator():
    # 示例数据：每批次生成5个样本，每个样本有100个特征
    rng = np.random.RandomState(1)
    while True:
        X_batch = rng.randint(5, size=(5, 100))
        y_batch = rng.randint(2, size=(5,))
        yield X_batch, y_batch

# 初始化分类器
clf = MultinomialNB()

# 获取数据生成器
generator = data_generator()

# 定义所有可能的类别
classes = np.array([0, 1])

# 增量训练模型
for _ in range(10):  # 假设我们有10个批次
    X_batch, y_batch = next(generator)
    clf.partial_fit(X_batch, y_batch, classes=classes)

# 在新数据上进行预测
X_new = np.random.randint(5, size=(1, 100))
prediction = clf.predict(X_new)
print("Prediction:", prediction)
```



sklearn 中支持增量学习的算法：

```
        Classification
                sklearn.naive_bayes.MultinomialNB
                sklearn.naive_bayes.BernoulliNB
                sklearn.linear_model.Perceptron
                sklearn.linear_model.SGDClassifier
                sklearn.linear_model.PassiveAggressiveClassifier
                sklearn.neural_network.MLPClassifier

        Regression
                sklearn.linear_model.SGDRegressor
                sklearn.linear_model.PassiveAggressiveRegressor
                sklearn.neural_network.MLPRegressor

        Clustering
                sklearn.cluster.MiniBatchKMeans
                sklearn.cluster.Birch

        Decomposition / feature Extraction
                sklearn.decomposition.MiniBatchDictionaryLearning
                sklearn.decomposition.IncrementalPCA
                sklearn.decomposition.LatentDirichletAllocation

        Preprocessing
                sklearn.preprocessing.StandardScaler
                sklearn.preprocessing.MinMaxScaler
                sklearn.preprocessing.MaxAbsScaler
```



# 其他模型



### 赋权法

赋权法用于需要 确定权重系数的情况。分为主观赋权法和客观赋权法。

熵权法 是客观赋权法的一种，层次分析法是主观赋权法的一种。



#### 熵权法

熵权法 计算  **信息熵**  来确定各个变量  **权重**  ，之后通过**变量和权重**可以计算出该条样本的  **评分**，用于挑选样本。

推导过程：https://www.zhihu.com/question/357680646/answer/943628631

```python
import pandas as pd
import numpy as np
import math
from numpy import array

df = pd.read_csv('aaa.csv')

#该方法要求x全为数值型
def cal_weight(x):
    x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))
    # 求k
    rows = x.index.size  # 行
    cols = x.columns.size  # 列

    k = 1.0 / math.log(rows)

    lnf = [[None] * cols for i in range(rows)]
    # 矩阵计算--
    # 信息熵
    # p=array(p)
    x = array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = array(lnf)

    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf)
    E = lnf

    # 计算冗余度
    d = 1 - E.sum(axis=0)
    # 计算各指标的权重
    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj
        # 计算各样本的综合得分,用最原始的数据

    w = pd.DataFrame(w)
    return w

w = cal_weight(df)
w.index = df.columns
w.columns = ['weight']

df['score'] = 0
for i in df.columns:
    if i != 'score':
        df['score'] = df['score'] + df[i]*w.loc[i,'weight']

# w 是每个特征的权重，df['score'] 是每条数据的得分
```



#### 层次分析法

层次分析法本质上通过人的主观判断给变量不同的权重，然后再通过验证，最终确定权重。确定权重之后即可给每条数据打分。

使用流程：https://blog.csdn.net/lengxiao1993/article/details/19575261

```python
import numpy as np
import pandas as pd

# # 需要人手动给出比较矩阵
# compare_matrix = np.array([[1, 0.2, 0.33, 1],
#                           [5, 1, 1.66, 5],
#                           [3, 0.6, 1, 3],
#                           [1, 0.2, 0.33, 1]])

# # 一致性检验
# def isConsist(F):
#     n = np.shape(F)[0]
#     a, b = np.linalg.eig(F)
#     maxlam = a[0].real
#     CI = (maxlam - n) / (n - 1)
#     if CI < 0.1:
#         return bool(1)
#     else:
#         return bool(0)
# # 一致性检验异常
# class isConsistError(Exception):
#     def __init__(self, value):
#         self.value = value

#     def __str__(self):
#         return repr(self.value)

# if isConsist(compare_matrix) == False:
#     raise isConsistError('一致性检验不通过')

# # 根据相对矩阵，算出每一个影响因素的重要程度
# def ReImpo(F):
#     n = np.shape(F)[0]
#     W = np.zeros([1, n])
#     for i in range(n):
#         t = 1
#         for j in range(n):
#             t = F[i, j] * t
#         W[0, i] = t ** (1 / n)
#     W = W / sum(W[0, :])  # 归一化 W=[0.874,2.467,0.464]
#     return W.T
# W = ReImpo(compare_matrix)


# #此矩阵的每一行代表一个方案，每一列代表一个影响因素/此矩阵为方案层权重矩阵，大部分场景可以自动生成
# df = pd.read_csv('entropy_weight.csv')

# #归一化方案层权重矩阵
# for i in df:
#     df[i] = df[i]/sum(df[i])
    
    
# score = np.dot(df,W)
# score = pd.DataFrame(score)
# print(score)
```



# 模型评价

模型指标评价包括业务（在线）指标评价，和离线指标（分类，回归，聚类等）评价。模型衰减也是一个评价维度。



### 业务指标

根据具体的业务制定线上指标，如点击率，转化率。



### 分类指标

使用分类指标对对分类模型进行评价。



##### confusion_matrix	

| 实际\预测           | 预测正类 (Positive) | 预测负类 (Negative) |
| ------------------- | ------------------- | ------------------- |
| 实际正类 (Positive) | TP (True Positive)  | FN (False Negative) |
| 实际负类 (Negative) | FP (False Positive) | TN (True Negative)  |



```python
from sklearn.metrics import confusion_matrix

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]

confusion_matrix(y_true, y_pred)
```



```python
# 多标签混淆矩阵
from sklearn.metrics import multilabel_confusion_matrix

y_true = np.array([[0, 0, 1],[0, 1, 0],[1, 1, 0]])
y_pred = np.array([[0, 1, 0],[0, 0, 1],[1, 1, 0]])

mcm = multilabel_confusion_matrix(y_true, y_pred)

tn = mcm[:, 0, 0]
tp = mcm[:, 1, 1]
fn = mcm[:, 1, 0]
fp = mcm[:, 0, 1]
```



##### 分类报告

```python
from sklearn.metrics import classification_report

y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 1, 0]

classification_report(y_true, y_pred)
```



##### accuracy

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

```python
from sklearn.metrics import accuracy_score

y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]

accuracy_score(y_true, y_pred)
```



##### precision

![3771db7af1e3b7bf33e15ec20d278f39](https://raw.githubusercontent.com/zhanghongyang42/images/main/3771db7af1e3b7bf33e15ec20d278f39.png)

```python
from sklearn.metrics import precision_score

y_pred = [0, 1, 0, 0]
y_true = [0, 1, 0, 1]

precision_score(y_true, y_pred,average='macro')
```



##### recall

![407341c3d4d055b857bb3229003b9daf](https://raw.githubusercontent.com/zhanghongyang42/images/main/407341c3d4d055b857bb3229003b9daf.png)

```python
from sklearn.metrics import recall_score

y_pred = [0, 1, 0, 0]
y_true = [0, 1, 0, 1]

recall_score(y_true, y_pred,average='macro')
```



##### F1

F1 的 β 为 1。

![b3edbb24837112f795a22e3574457416](https://raw.githubusercontent.com/zhanghongyang42/images/main/b3edbb24837112f795a22e3574457416.png)

```python
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score

y_pred = [0, 1, 0, 0]
y_true = [0, 1, 0, 1]

f1_score(y_true, y_pred,average='macro')
fbeta_score(y_true, y_pred, beta=2)
```



##### 精确率-召回率曲线

```python
#计算出不同阈值下的精确率和召回率
from sklearn.metrics import precision_recall_curve
precision, recall, threshold = precision_recall_curve(y_true, y_scores)

# 绘制精确率-召回率曲线
plt.plot(recall, precision, marker='.')
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('精确率-召回率曲线')
plt.show()
```



##### ROC曲线

ROC（Receiver Operating Characteristic）曲线用于评估二分类模型的性能。横轴是不同阈值下的 **假阳性率 (FPR)**，纵轴是不同阈值下的 **真正例率 (TPR)**：
$$
TPR = \frac{TP}{TP + FN}
$$

$$
FPR = \frac{FP}{FP + TN}
$$



![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/v2-66cc29b9e9f951de48c214d9ec34f4c5_1440w.jpg)

一个好的分类器应当在尽可能低的 FPR 下获得较高的 TPR，从而使 ROC 曲线尽可能靠近左上角（理想点）。此时 AUC 也是尽可能接近 1的。



------

ROC曲线 绘制

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 假设 y_true 是真实标签，y_scores 是预测概率
y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.6, 0.3, 0.5])

# 计算 FPR, TPR 和 阈值
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# 计算 AUC（曲线下的面积）
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC 曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 随机猜测的参考线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (False Positive Rate)')
plt.ylabel('真正例率 (True Positive Rate)')
plt.title('ROC 曲线')
plt.legend(loc="lower right")
plt.show()
```



##### AUC

**几何角度定义**：AUC（曲线下面积，Area Under the Curve）是 ROC 曲线下的面积，使用积分计算。

**概率角度定义**：AUC 是随机选一个正样本和一个负样本，正样本预测值大于负样本预测值的概率（即排序正确的比例）。

可通过计算，这两个定义会得到同一个AUC。



- **AUC ≈ 1.0** → 说明分类效果非常好。
- **AUC ≈ 0.5** → 说明模型和随机猜测没区别（无信息）。
- **AUC < 0.5** → 说明模型可能预测方向反了（需要调整概率阈值）。



**阈值无关性**：
ROC 曲线和 AUC 是阈值无关的评价指标。与依赖特定阈值（如准确率、精确率）的指标不同，ROC 曲线综合了所有可能的决策阈值，评估了模型的整体性能。

**平衡正负样本**：
无论正负样本数量是否平衡，ROC 曲线和 AUC 都能反映模型对正负样本的区分能力，这在不平衡数据场景下尤其重要。

**排序能力的量化**：
由于 AUC 可以解释为正样本得分高于负样本得分的概率，因此它不仅评估了分类决策，还定量反映了模型输出分数的排序效果。



tips：max AUC 由数据决定，一种极端情况，两条数据，特征一样，label不同，AUC 最大是 0.5。可以通过预测 训练集数据来大致看一下max AUC。

------

AUC 计算代码

```python
import numpy as np
from sklearn.metrics import roc_auc_score

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])

roc_auc_score(y_true, y_scores)
```



```python
#计算多分类AUC得分

roc_auc_score(y_true, y_score, multi_class='ovr')  # 一对多
roc_auc_score(y_true, y_score, multi_class='ovo')  # 一对一
```



### 回归指标

使用回归指标对回归模型进行评价。



##### MSE

均方误差，Mean Squared Error


$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$


```python
from sklearn.metrics import mean_squared_error

mean_squared_error(y_true, y_pred)

#多标签
mean_squared_error(y_true, y_pred)  
```



##### RMSE（常用）

均方根误差，Root Mean Squared Error


$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$


```python
from sklearn.metrics import mean_squared_error, r2_score

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("RMSE:", rmse(y_true, y_pred))
```



##### MAE

Mean Absolute Error，平均绝对误差


$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$


```python
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_true, y_pred)

#多标签
mean_absolute_error(y_true, y_pred)
```



##### MAPE（常用）

平均绝对百分比误差，Mean Absolute Percentage Error

注意，真实值接近0时，这个指标不太可用。


$$
MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%
$$


```python
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("MAPE:", mape(y_true, y_pred))
```



##### R²（常用）

决定系数，R-squared

R² 越接近 1，说明模型越好地拟合了数据。R² = 0，模型没有任何预测能力。


$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$


```python
from sklearn.metrics import r2_score

r2_score(y_true, y_pred)
```



##### MedAE

中位数绝对误差，Median Absolute Error


$$
MedAE = median(|y₁ - ŷ₁|, |y₂ - ŷ₂|, ..., |yₙ - ŷₙ|)
$$


```python
from sklearn.metrics import median_absolute_error

median_absolute_error(y_true, y_pred)
```



### 聚类指标



##### 轮廓系数

![8f839ebe5b506fef19bd8cc121b3f557](https://raw.githubusercontent.com/zhanghongyang42/images/main/8f839ebe5b506fef19bd8cc121b3f557.png)

- **a**: 样本与同一类别中所有其他点之间的平均距离。
- **b**: 样本与 下一个距离最近的簇 中的所有其他点之间的平均距离。

```python
from sklearn.cluster import KMeans
from sklearn.silhouette_score import silhouette_score

kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_

silhouette_score(X, labels, metric='euclidean')
```



##### 调整兰德系数

Adjusted Rand Index, ARI

已知真实标签时 对聚类算法的评价。



兰德系数是把聚类当成分类求 acc，调整兰德系数是在这个基础上，进行了调整。

------



真实标签和聚类结果的列联表如下：

|            | $V_1 $    | $V_2 $    | ...  | $V_c $    | **行和** |
| ---------- | --------- | --------- | ---- | --------- | -------- |
| **$U_1$**  | $n_{11} $ | $n_{12} $ | ...  | $n_{1c} $ | $a_1 $   |
| **$U_2 $** | $n_{21} $ | $n_{22} $ | ...  | $n_{2c} $ | $a_2 $   |
| ...        | ...       | ...       | ...  | ...       | ...      |
| **$U_r $** | $n_{r1} $ | $n_{r2} $ | ...  | $n_{rc} $ | $a_r $   |
| **列和**   | $b_1 $    | $b_2 $    | ...  | $b_c $    | $n $     |

- $U = \{U_1, U_2, \ldots, U_r\} $：真实标签的分组（共 $ r $个组）。

- $V = \{V_1, V_2, \ldots, V_c\} $：聚类结果的分组（共 $ c $个组）。

- n ：总样本数。

- $n_{ij} $：同时属于真实组 $U_i$和聚类组$ V_j $的样本数（即$U_i \cap V_j$的样本数）。

  

ARI 的计算公式为：

$$
ARI = \frac{ \sum{ij} \binom{n{ij}}{2} - \frac{ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} }{ \binom{n}{2} } }{ \frac{ \sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2} }{2} - \frac{ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} }{ \binom{n}{2} } }
$$

- $\binom{n_{ij}}{2} = \frac{n_{ij}(n_{ij}-1)}{2} $表示在真实组 $ U_i $和聚类组 $ V_j $中一致的对数。
- $\binom{a_i}{2} = \frac{a_i(a_i-1)}{2} $表示真实组 $ U_i $中所有可能的样本对数。
- $\binom{b_j}{2} = \frac{b_j(b_j-1)}{2} $表示聚类组 $ V_j $中所有可能的样本对数。
- $\binom{n}{2} = \frac{n(n-1)}{2} $表示所有样本的总对数。

------



```python
from sklearn.metrics import adjusted_rand_score

# 示例：真实标签和聚类结果
true_labels = [0, 0, 1, 1, 1]
cluster_labels = [0, 0, 1, 2, 2]

ari = adjusted_rand_score(true_labels, cluster_labels)
print("ARI:", ari)
```



- ARI 的取值范围为 [-1, 1]：
  - **ARI = 1**：聚类结果与真实标签完全一致。
  - **ARI = 0**：聚类结果与随机分配相当。
  - **ARI < 0**：聚类结果比随机分配更差（通常不会出现，除非聚类算法刻意反向聚类）。



### 模型衰减

模型衰减可以通过训练集和测试集指标之间的差值来 反应模型衰减性能。

差值小，说明泛化性能好，说明模型衰减慢。

如下面举例，B模型效果差一些，但是泛化能力更好，衰减更慢。

| 模型 | 训练集AUC | 测试集AUC | 差值 |
| ---- | --------- | --------- | ---- |
| A    | 0.9       | 0.8       | 0.1  |
| B    | 0.8       | 0.75      | 0.05 |



# 交叉验证

把模型直接划分为训练集和测试集查看模型效果，可能会因为数据划分，产生不准确的模型评价。

所以模型评估时，把数据多次划分为训练集和测试集。平均多次测试集的结果，可以使模型评估更准确。

![img](https://raw.githubusercontent.com/zhanghongyang42/images/main/v2-7f165ecd9559047847a04342df538ea0_r.jpg)



```python
from sklearn.datasets import load_iris
iris = load_iris()

from sklearn import svm
clf = svm.SVC(kernel='linear', C=1)

# 交叉验证
from sklearn.model_selection import cross_validate
scores = cross_validate(clf, iris.data, iris.target, cv=5, scoring='f1_macro',return_estimator=True)

print(scores.keys())
print(scores['fit_time'])
print(scores['score_time'])
print(scores['estimator'])
print('Accuracy : %0.2f'%scores['test_score'].mean())

# k折交叉验证、分层k折(用于数据不平衡)、自定义交叉验证
from sklearn.model_selection import KFold,StratifiedKFold,ShuffleSplit
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

#按时间交叉验证
https://zhuanlan.zhihu.com/p/99674163
```



# 超参搜索

模型有多个超参数，每个超参数有合适的取值，使用超参数搜索为模型选择合适的超参数。



超参数搜索的步骤是，多组超参数，交叉验证。就得到了最好的超参数，最好的模型和模型评价。

超参数搜索搜索完成后，使用全部数据的训练集测试集 进行正常的模型训练，模型评价即可。也可以直接使用超参数搜索中表现最好的那组模型。



超参数搜索中，每组超参数上使用交叉验证，防止了某组超参数因为数据划分表现更好。也就是防止了过拟合（防止了这组超参数在测试集上表现不好）。



### 网格搜索

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold

p_grid = {"C": [1, 10, 100],"gamma": [.01, .1]}
svm = SVC(kernel="rbf")
cv = KFold(n_splits=4, shuffle=True, random_state=i)
 
clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=cv)
clf.fit(X_iris, y_iris)

print(clf.best_score_)
print(clf.best_params_)
clf = clf.best_estimator_
```



### 随机搜索

相比于网格搜索，相当于用一点准确性换来了大幅的性能提升。

```python
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

clf = RandomForestClassifier(n_estimators=20)
param_dist = {"max_depth": [3, 4],"max_features": sp_randint(1, 11),"criterion": ["gini", "entropy"]}

random_search = RandomizedSearchCV(clf, param_distributions=param_dist,n_iter=20, cv=5, iid=False)
random_search.fit(X, y)
```



### 网格搜索评价

#### 现有的*网格得分*函数

| Scoring（得分）                | Function（函数）                                             | Comment（注解）                                              |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Classification（分类）**     |                                                              |                                                              |
| ‘accuracy’                     | [`metrics.accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score) |                                                              |
| ‘average_precision’            | [`metrics.average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score) |                                                              |
| ‘f1’                           | [`metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) | for binary targets（用于二进制目标）                         |
| ‘f1_micro’                     | [`metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) | micro-averaged（微平均）                                     |
| ‘f1_macro’                     | [`metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) | macro-averaged（宏平均）                                     |
| ‘f1_weighted’                  | [`metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) | weighted average（加权平均）                                 |
| ‘f1_samples’                   | [`metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) | by multilabel sample（通过 multilabel 样本）                 |
| ‘neg_log_loss’                 | [`metrics.log_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss) | requires `predict_proba` support（需要 `predict_proba` 支持） |
| ‘precision’ etc.               | [`metrics.precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score) | suffixes apply as with ‘f1’（后缀适用于 ‘f1’）               |
| ‘recall’ etc.                  | [`metrics.recall_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score) | suffixes apply as with ‘f1’（后缀适用于 ‘f1’）               |
| ‘roc_auc’                      | [`metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score) |                                                              |
| **Clustering（聚类）**         |                                                              |                                                              |
| ‘adjusted_mutual_info_score’   | [`metrics.adjusted_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score) |                                                              |
| ‘adjusted_rand_score’          | [`metrics.adjusted_rand_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score) |                                                              |
| ‘completeness_score’           | [`metrics.completeness_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score) |                                                              |
| ‘fowlkes_mallows_score’        | [`metrics.fowlkes_mallows_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html#sklearn.metrics.fowlkes_mallows_score) |                                                              |
| ‘homogeneity_score’            | [`metrics.homogeneity_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score) |                                                              |
| ‘mutual_info_score’            | [`metrics.mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html#sklearn.metrics.mutual_info_score) |                                                              |
| ‘normalized_mutual_info_score’ | [`metrics.normalized_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score) |                                                              |
| ‘v_measure_score’              | [`metrics.v_measure_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score) |                                                              |
| **Regression（回归）**         |                                                              |                                                              |
| ‘explained_variance’           | [`metrics.explained_variance_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score) |                                                              |
| ‘neg_mean_absolute_error’      | [`metrics.mean_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error) |                                                              |
| ‘neg_mean_squared_error’       | [`metrics.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error) |                                                              |
| ‘neg_mean_squared_log_error’   | [`metrics.mean_squared_log_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error) |                                                              |
| ‘neg_median_absolute_error’    | [`metrics.median_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error) |                                                              |
| ‘r2’                           | [`metrics.r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score) |                                                              |

```python
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
X, y = iris.data, iris.target
clf = svm.SVC(probability=True, random_state=0)
cross_val_score(clf, X, y, scoring='neg_log_loss')
```



#### 构造*网格得分*函数

##### 从metrics构造

```
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

ftwo_scorer = make_scorer(fbeta_score, beta=2)

grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=ftwo_scorer)
```



##### 自定义评价函数 构造

```python
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.dummy import DummyClassifier

#构建了一个评分函数
def my_custom_loss_func(y_true, y_pred):
    diff = np.abs(y_true - y_pred).max()
    return np.log1p(diff)

#数据集
X = [[1], [1]]
y = [0, 1]

#训练模型
clf = DummyClassifier(strategy='most_frequent', random_state=0)
clf = clf.fit(X, y)

#进行评价
print(my_custom_loss_func(clf.predict(X), y))

#构造网格得分函数
score = make_scorer(my_custom_loss_func, greater_is_better=False)
#greater_is_better返回的是损失得分，即函数返回值的相反数
print(score(clf, X, y))
```



# 自定义估计器

如果有些特征工程和算法，sklearn没有提供，我们又想使用的话，可以使用自定义的方法。



### FunctionTransformer

自定义简单的Transformer，如数值计算，类型转换等。

```python
def tofloat(x):
    return x.astype(np.float64)

tofloat_transformer = FunctionTransformer(tofloat, accept_sparse=True)
```

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p)
transformer.transform(X)
```

注意，如果使用 lambda  表达式自定义方法，FunctionTransformer不能被pickle



### 自定义 Transformer 

自定义 Transformer 和 Estimator，本部分内容仅供参考，还需整理。

```python
#异常值处理
#指定值替换
class ValueReplace(BaseEstimator,TransformerMixin):
    def __init__(self,column,orig_value,rep_value):
        self.column = column
        self.orig_value = orig_value
        self.rep_value = rep_value
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X[self.column].replace(self.orig_value,self.rep_value,inplace=True)
        return X
    
#特征衍生
#经验手动衍生
class FeatureDivision(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X["zd_bal_rat"] = X["zd_bal"]/X["acct_cred_limit"]
        X["in_bal_rat"] = X["in_bill_bal"]/X["acct_cred_limit"]
        X["all_bal_rat"] = (X["in_bill_bal"]+X["zd_bal"])/X["acct_cred_limit"]
        X["mean_deposit_3"] = (X["deposit"]+X["lm_deposit"]+X["lm2_deposit"])/3 
        X["deposit_bal_rat"] = X["deposit"]/X["zd_bal"]
        X["deposit_bal_bill_rat"] = X["deposit"]/(X["zd_bal"]+X["in_bill_bal"])
        return X
    def fit_transform(self,X,y=None):
```



# pipeline

pipeline 把多个 模型&特征工程 组合成一个模型。便于上线调用。

pipeline 和 模型一样，可以训练、预测、评价、超参数搜索、持久化、部署。



下面有3个类用于转换X,，分别是**Pipeline、FeatureUnion、ColumnTransformer**，这三个类可以互相嵌套，用于设计pipeline。

`Pipeline串行处理所有特征，FeatureUnion并行处理所有特征，ColumnTransformer并行处理选择的特征`



**TransformedTargetRegressor** 用于转换y，详见 https://scikit-learn.org/stable/modules/compose.html



**tips**：pipeline 除了最后一个estimator 可以是只继承estimator的算法外，其他estimator 必须是继承estimator 和 transformer 的特征工程。

**tips**：当sklearn的estimator不能满足我们的需要时，简单计算可以使用 FunctionTransformer，复杂计算需要自定义 Transformer 和 Estimator，详见sklearn



### Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

pipe = Pipeline([('reduce_dim', PCA()), ('clf', SVC())])

# 查询
pipe.steps 
pipe['reduce_dim']

# 删除或者替换模型
pipe.set_params(clf='drop')

# 设置参数
pipe.set_params(clf_C=10) #clf 是我们在pipeline中设置的评估器名称，C 是对应的参数名称
```



网格搜索

```python
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from tempfile import mkdtemp

pipe = Pipeline([('reduce_dim', PCA()), ('clf', SVC())], memory=mkdtemp()) #开启缓存，同样参数的中间transformer不会重新训练
param_grid = dict(reduce_dim_n_components=[2, 5, 10],clf_C=[0.1, 10, 100])

grid_search = GridSearchCV(pipe, param_grid=param_grid)
```



### FeatureUnion

把多个transformer 并联成一个transformer，每一个transformer 都要处理所有的输入数据，然后把他们的输出组合。

```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

combined = FeatureUnion([('linear_pca', PCA()), ('kernel_pca', KernelPCA())])
```



### ColumnTransformer

把不同的列用不同的方法处理

```python
import pandas as pd
X = pd.DataFrame({
    'city': ['London', 'London', 'Paris', 'Sallisaw'],
     'title': ["His Last Bow", "How Watson Learned the Trick","A Moveable Feast", "The Grapes of Wrath"],
     'expert_rating': [5, 3, 4, 5],
     'user_rating': [4, 5, 4, 3]
    })

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

column_trans = ColumnTransformer(
    [
        ('city_category', OneHotEncoder(dtype='int'),['city']),
        ('title_bow', CountVectorizer(), 'title')
    ],
    remainder=MinMaxScaler()) #remainder='drop',remainder='passthrough',verbose_feature_names_out=False

column_trans.fit_transform(X)
```



### 自定义 FeatureUnion

Pipeline、FeatureUnion、ColumnTransformer 返回的都是estimator ，如果想要返回值是 dataframe，需要自定义。一般没有这种需求。

```python
class PandasFeatureUnion(FeatureUnion):
    def fit_transform(self,X,y=None,**fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)
        (delayed(_fit_transform_one)(
            transformer=trans,
            X=X,
            y=y,
            weight=weight,
            **fit_params)
         for name,trans,weight in self._iter())
        if not result:
            return np.zeros(x.shape[0],0)
        Xs,transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs
    
    def merge_dataframes_by_column(self,Xs):
        return pd.concat(Xs,axis='columns',copy=False)
    
    def transform(self,X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name,trans,weight in self._iter())
        if not Xs:
            return np.zeros((x.shape[0],0))
        if any(sparse.issparse(f) for f in Xs):
            Xs=sparse.hstack(Xs).tocsr()
        else:
            Xs=self.merge_dataframes_by_column(Xs)
        return Xs
```



# 模型持久化

### pickle

python中几乎所有的数据类型（列表，字典，集合，类等）都可以用  pickle  来序列化

```python
import pickle

with open('svm.pickle','wb') as fw:
pickle.dump(svm,fw)				#文件保存
with open('svm.pickle', 'rb') as fr:
svm = pickle.load(fr)		#文件加载

s = pickle.dumps(clf)			#内存保存
clf = pickle.loads(s)			#内存加载
```



### joblib

一般模型都可以用  joblib  来序列化

```python
from sklearn.externals import joblib

joblib.dump(clf, 'filename.pkl')
clf = joblib.load('filename.pkl')
```



### PMML

官网：https://github.com/jpmml?page=1

比较重要的项目包括：[sklearn2pmml](https://github.com/jpmml/sklearn2pmml)、[jpmml-sklearn](https://github.com/jpmml/jpmml-sklearn)、[sklearn2pmml-plugin](https://github.com/jpmml/sklearn2pmml-plugin)

介绍：https://www.cnblogs.com/pinard/p/9220199.html



#### 简介

pmml 是一种跨多语言平台（包括java和python）的序列化格式。可以用来跨语言环境部署模型。

但是这种格式对模型来说有一定的精度损失。

pmml本质上就是将模型序列化成的XML文件。

实际上线时，上百维的结构化数据可以使用pmml这种方式上线。



#### 使用

pmml 一般使用方法如下，直接构建好一个pipeline（见上一节pipeline），然后转换成pmml_pipeline,直接保存。

```python
# 保存
from sklearn2pmml import make_pmml_pipeline,sklearn2pmml

pipeline_pmml = make_pmml_pipeline(pipeline)
sklearn2pmml(pipeline_pmml, "./model.pmml", with_repr = True, debug = True)

#加载，一般用java
```



pipeline能支持的transformer很多，详见上一节 pipeline。

但是pmml支持的transformer不多，详见 https://github.com/jpmml/jpmml-sklearn#supported-packages 。



还有pmml关于一些非sklearn标准库使用的方法示例：https://github.com/jpmml/sklearn2pmml#documentation ，如改变数据域防止java调用报错。

更多pmml自己支持的方法，详见源码：https://github.com/jpmml/sklearn2pmml/tree/master/sklearn2pmml



#### 自定义

若是sklearn标准库中没有，pmml也没有补充的数据处理方法，只能自己自定义。

自定义方法如下https://github.com/jpmml/sklearn2pmml-plugin



自定义原理是：

sklearn2pmml 中的每一个类，在java中都有一个对应的类去实现操作。

通过把python中类对象 名称和参数 序列化到 xml 文件中，java 去解析 xml 文件。

如果想要自定义新的transformer，只能按照上述方法，在python端和java端分别实现一个进行xml文件的序列化和解析。

任何在java端没有对应名称的方法，都无法被解析。



# 模型部署

https://www.zhihu.com/question/37426733/answer/786472048

https://my.oschina.net/taogang/blog/2222908



# 模型上线

### 线上指标

模型线上指标 根据场景的不同有所不同。

如搜索排序场景下，可以使用 点击数/曝光数 计算点击率来作为线上指标。也可以用其他转化率作为线上指标。

也可以使用 点击位置，翻页次数等个性化的指标作为辅助。



线上最主要的指标可能和模型训练的 label 是一致的。

线上指标最好可以实时监控。



### 上线标准

首个模型第一次上线：模型满足 离线指标 如 auc 大于 0.8 后可以上线。

同一个模型，为了防止时间衰减进行更新：

​	1.简单的，可以过一段时间重新训练一个新模型进行更新

​	2.复杂的，可以进行线上指标监控，当线上指标下降的时候，训练一个新模型，看离线AUC，在一定范围内可以上线。

不同的模型，可以在达到离线指标后，进行ABtest上线，看线上指标。



模型离线指标与在线指标变化不一致：https://zhuanlan.zhihu.com/p/443208809



# 建模示例

Titanic 生存预测。

这个示例手动的进行了模型的融合。

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
pd.options.display.max_columns = None

#读入数据
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#查看数据是否平衡
print(train_data["Survived"].value_counts())

#查看数值型还是类别型
print(train_data.dtypes)

#查看缺失值，结合分析出异常值，0值，查看高基数特征，整体相关性
# if __name__ == '__main__':
#     pfr = pandas_profiling.ProfileReport(train_data)
#     pfr.to_file("./example_train.html")
#     pfr = pandas_profiling.ProfileReport(test_data)
#     pfr.to_file("./example_test.html")

#缺失值处理
train_data = train_data.drop(['Cabin'],axis=1)

#数值化
train_data = train_data.drop(['Name','Ticket'],axis=1)
train_onehot = pd.get_dummies(train_data[['Sex','Embarked']])

train_data = pd.concat([train_data,train_onehot],axis=1)
train_data = train_data.drop(['Sex','Embarked'],axis=1)

#只对数值型起作用，0.1以下认为没有相关性，0.5以上认为强相关
correlations = train_data.corr()['Survived'].sort_values()
print(correlations)
#进一步查看与label的相关性
def kde_target(df, lable, var_name):
    plt.figure(figsize = (12, 6))
    sns.kdeplot(df.ix[df[lable] == 0, var_name], label = '0')
    sns.kdeplot(df.ix[df[lable] == 1, var_name], label = '1')
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend()#加上图例
    plt.show()
kde_target(train_data,"Survived","Parch")
kde_target(train_data,"Survived","SibSp")
kde_target(train_data,"Survived","PassengerId")
kde_target(train_data,"Survived","Age")
kde_target(train_data,"Survived","Embarked_Q")

train_data = train_data.drop(['Age','Embarked_Q','PassengerId'],axis=1)

#删除线性相关变量，看热力图
def test(df):
    dfData = df.corr()
    plt.subplots(figsize=(9, 9)) # 设置画面大小
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
    plt.show()
test(train_data)

#选出x和y
x = train_data.drop(["Survived"],axis=1)
y = train_data['Survived']

#切分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

#模型融合
    # 导入一级模型
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier)
import xgboost as xgb
from sklearn.svm import SVC
rf_model = RandomForestClassifier(random_state=42)
adb_model = AdaBoostClassifier(random_state=42)
gdbc_model = GradientBoostingClassifier(random_state=42)
et_model = ExtraTreesClassifier(random_state=42)
svc_model = SVC(random_state=42)
xgb_model = xgb.XGBClassifier(seed=42,max_depth=3,learning_rate=0.1,n_estimators=40,random_state=42)
    #模型预测方法
def get_stacking(clf, x_train, y_train, x_test, n_folds=10):
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_train = np.zeros((train_num,))
    second_test = np.zeros((test_num,))
    test_nfolds = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst = x_train[test_index], y_train[test_index]
        clf.fit(x_tra, y_tra)
        second_train[test_index] = clf.predict(x_tst)
        test_nfolds[:, i] = clf.predict(x_test)
        second_test[:] = test_nfolds.mean(axis=1)
    return second_train, second_test
    #存放所有模型的结果
train_sets = []
test_sets = []
for clf in [rf_model, adb_model, gdbc_model, et_model, svc_model , xgb_model]:
    train_set, test_set = get_stacking(clf, X_train.values, y_train.values, X_test.values)
    train_sets.append(train_set)
    test_sets.append(test_set)
    #把所有训练集结果按列合并
meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)
    #二级分类器
from xgboost import XGBClassifier
xgb = XGBClassifier( colsample_bytree=0.8,
                    gamma=5,
                    learning_rate=0.05,
                    max_depth=3,
                    min_child_weight=2,
                    n_estimators=600,
                    random_state=42,
                    reg_alpha=2,
                    reg_lambda=2,
                    subsample=0.75)
xgb.fit(meta_train, y_train)
y_score = xgb.predict(meta_train)
y_score_test = xgb.predict(meta_test)

# #搜索超参数
# import xgboost as xgb
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import GridSearchCV
# xgb_model = xgb.XGBClassifier(seed=42)
# max_depth = [3]
# learning_rate = [0.1]
# n_estimators = [40]
# param_grid = dict(max_depth = max_depth,learning_rate = learning_rate,n_estimators =n_estimators)
# kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#
# # 训练模型
# grid_search = GridSearchCV(xgb_model,param_grid,cv = kflod)
# grid_search = grid_search.fit(X_train, y_train)
# xgb_model = grid_search.best_estimator_
# print(grid_search.best_params_)
#
# pred_test = xgb_model.predict(X_test)

#预测验证集上的得分
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_score_test))
print(metrics.accuracy_score(y_train, y_score))

#对测试集也做相关操作
test_data = test_data.drop(['Cabin','Name','Ticket'],axis=1)
test_onehot = pd.get_dummies(test_data[['Sex','Embarked']])
test_data = pd.concat([test_data,test_onehot],axis=1)
PassengerId = test_data['PassengerId']
test_data = test_data.drop(['Sex','Embarked','Age','Embarked_Q','PassengerId'],axis=1)

    #模型融合，一级模型预测
test_data = test_data.fillna(0)
sets = []
for clf in [rf_model, adb_model, gdbc_model, et_model, svc_model , xgb_model]:
    set = clf.predict(test_data.values)
    sets.append(set)
data = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in sets], axis=1)

#测试集预测保存
predictions = xgb.predict(data)
output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

#基本就到这里了。优化先做特征，再搞模型 ，后续优化思路：模型融合的一级模型参数可以调整，二级分类器参数也可以搜索，深度挖掘特征（比如Name中的姓氏，title。单身和家庭情况。或者对丢弃字段做一个填充，比如年龄，进行模型填充，离散化）
```



# 工程优化

### 内存优化

python 一般是单机运行，在大数据量场景中很容易出现内存不足的情况，可以优化内存使用情况。



##### 删除变量

```python
del df
```



##### 持久化到磁盘

将读取或者处理过数据使用pickl持久化到硬盘，清空内存重新读取可以一次性清除一些不用的内存。



##### 学习曲线

多次运行的场景可以使用学习曲线 选出最优数据量，即使用一部分数据就可以达到90%的效果

```python
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)
```



### 耗时优化



##### 持久化到磁盘

将读取或者处理过数据使用pickl持久化到硬盘，避免重复运算



##### 指定数据类型

读取数据时，指定数据类型，会加快读取速度，代码仅供参考。

```python
import pandas as pd

#读入数据，查看数据类型
df = pd.read_csv('train.csv')
print(df.dtypes)

#对不同的数据类型进行不同的优化
for col in df.select_dtypes('object'):
    if len(df[col].unique())/len(df[col])<0.5:
        df[col] = df[col].astype('category')

aa = df.select_dtypes('int64').apply(pd.to_numeric,downcast='signed')
bb = df.select_dtypes('float').apply(pd.to_numeric,downcast='float')

#将优化后的数据类型存到字典中
dic = {}
for i,j in zip(df.dtypes,df.dtypes.index):
    dic[j] = i
for i,j in zip(aa.dtypes,aa.dtypes.index):
    dic[j] = i
for i,j in zip(bb.dtypes,bb.dtypes.index):
    dic[j] = i

#重新读取数据
del df,aa,bb
df = pd.read_csv('train.csv',dtype=dic)
```



# 模型优化

### 最优模型阈值

```python
# 可以通过调节阈值调整 召回率和准确率
y_prob = pd.DataFrame(clf.predict_proba(X_test)[:,1])

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

for i in range(9):
    i = (i+1)/10
    y_pred = y_prob.applymap(lambda x: 1 if x>=i else 0)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(roc_auc_score(y_test, y_pred))
    print('---------------------------------------------------')
```















