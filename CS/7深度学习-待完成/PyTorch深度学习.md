动手学深度学习 PyTorch 视频：https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497

动手学深度学习：https://zh-v2.d2l.ai/



# pytorch 安装

PyTorch 论坛 ：https://discuss.pytorch.org/



### Mac & Linux

https://zh-v2.d2l.ai/chapter_installation/index.html 。



### windows (CPU)



安装 Anaconda



创建新环境

```
conda env remove -n d2l-zh

conda create -n d2l-zh python=3.8 pip -y

conda activate d2l-zh
```



cmd 查看 cuda 版本

```
nvidia-smi
```



进入 https://pytorch.org/，获取命令

```
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install d2l
```



验证，在新环境安装  jupyter notebook

```python
import torch
print(torch.rand(5,3))

# true
torch.cuda.is_available() 
```



文档查询

```python
# 查询某类的属性
print(dir(torch.distributions))

# 查看说明
help(torch.ones)
```



### 代码环境

```
conda activate d2l-zh

D:
cd D:\PythonProject
mkdir d2l-zh && cd d2l-zh

curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
(手动) unzip d2l-zh.zip && rm d2l-zh.zip

jupyter notebook
```



# 预备知识

n维数组，也称为*张量*（tensor）。*张量类*（在MXNet中为`ndarray`， 在PyTorch和TensorFlow中为`Tensor`）类似于 Numpy的`ndarray`。



### 张量使用

```python
import torch

x = torch.arange(12) #创建一个行向量 x，张量中的每一个值叫做元素。
X = torch.zeros((2, 3, 4))
X = torch.ones((2, 3, 4))
X = torch.randn(3, 4) 		#均值为0、标准差为1的标准高斯分布（正态分布）
X = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

x.numel()
x.shape
X = x.reshape(3,-1)
```

张量索引

```python
X[-1]
X[1:3]
X[1,2] = 9
X[0:2, :] = 12
```

简单运算，简单函数

```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
a = 2

# 两个张量对应元素运算
x + y, x - y, x * y, x / y, x ** y ,torch.exp(x)
a * x

#其他运算
X == Y
X.sum()
A.sum(axis=0) #求各行的和
A.sum(axis=[0, 1]) #沿轴求和
A.sum(axis=1, keepdims=True) #沿轴求和，维度不变
A.mean(axis=0)
A.cumsum(axis=0)
```

张量拼接

```python
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

torch.cat((X, Y), dim=0) # 按行拼接
torch.cat((X, Y), dim=1) # 按列拼接
```

小技巧

```python
# 广播机制 ，形状不匹配数组，第二个数组自动复制元素补齐，然后再按元素计算
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a + b

# 减少内存消耗，如果 X = X + Y，新的x会被分配新的内存
X[:] = X + Y

# 深度复制
B = A.clone()
```

类型转换

```python
X = torch.tensor(df.values)
A = X.numpy()
B = torch.tensor(A)

a = torch.tensor([3.5])
a.item()
float(a)
int(a)
```



### 线性代数

标量计算

```python
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

向量计算：

加减乘除、指数对数、三角函数 都是对应元素进行计算

```python
x + y, x - y, x * y, x / y, x ** y ,torch.exp(x)
```

矩阵计算

```python
# 点积
torch.dot(x, y)
torch.sum(x * y) 

# 矩阵*向量
torch.mv(A, x)

# 矩阵乘法
torch.mm(A, B)

# 范数，即距离
torch.norm(torch.ones((4, 9)))
```



### 微积分

微分就是求导，书写形式的区别。第一种形式是数值求导，第二种形式是符号求导
$$
f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.
$$

$$
f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),
$$



微分法则，求微分
$$
\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),
$$

$$
\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),
$$

$$
\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],
$$

$$
\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.
$$

快速求导公式：标量求导

![image-20250410171123207](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20250410171123207.png)



求偏导，求多个变量函数的导数，将其他变量看成常数，然后求导
$$
\frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.
$$

$$
\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.
$$



梯度，函数的所有偏导数组成的向量，也是向量对标量求导的结果
$$
\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,
$$


向量求导

![image-20250410172206813](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20250410172206813.png)



### 自动求导

因为链式法则的原因，自动求导有两种形式

![image-20250410201141547](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20250410201141547.png)

在计算机中，可以用有向无环图正向积累和反向积累的计算过程

![image-20250410201910832](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20250410201910832.png)



反向传播，不用计算中间结果，计算效率更高。



反向传播

```python
import torch

x = torch.arange(4.0,requires_grad=True) # 给变量x分配初始值，并存储梯度
y = 2 * torch.dot(x, x)

y.backward() 		# 反向传播计算梯度
x.grad == 4 * x 	# 函数y的梯度为4x
```

```python
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()

y = x.sum()
y.backward()
x.grad
```

非标量变量的反向传播

```python
# 先把向量y通过求和变成标量，再反向传播
y = x * x
y.sum().backward()
```

分离计算

```python
# 在PyTorch中，y.detach()的作用是从计算图中分离出张量y，生成一个与y共享数据但不携带梯度信息的新张量。分离后的张量不再参与梯度计算，从而阻断反向传播时梯度通过该节点的传递

y = x * x
u = y.detach() 
z = u * x

z.sum().backward()
x.grad == u
```



### 概论论

*样本空间*（sample space）,所有可能结果的集合。

*事件*（event），样本空间的一部分，一个可能的结果就是一个事件。

*概率*（probability）可以被认为是将事件集合映射到真实值的函数。 在给定的样本空间S中，事件A的概率， 表示为P(A)



概率论公理：

- 对于任意事件A，其概率从不会是负数，即P(A)≥0；
- 整个样本空间的概率为1，即P(S)=1；
- 对于*互斥*（mutually exclusive）事件的任意一个可数序列A1,A2,…，序列中任意一个事件发生的概率等于它们各自发生的概率之和；



随机变量 就是 P（X = 3）中的 X，随机变量可以是离散的也可以是连续的。

随机变量不同于代数中的变量，代数中使用的变量一次不能具有多个值。

如果随机变量X = {0,1,2,3} 那么X可以是随机的0、1、2或3，其中每个都有不同的概率。



*分布*（distribution）看作是对事件的概率分配 。分布的横轴是随机变量，纵轴是概率值。

*联合概率*（joint probability）P(A=a,B=b)。

*条件概率*（conditional probability）， 用P(B=b∣A=a)表示它：它是B=b的概率，前提是A=a已发生。

贝叶斯定理 ：因为 P(A,B)=P(B∣A)P(A)，P(A,B)=P(A∣B)P(B) 。所以
$$
P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.
$$
*边际概率*（marginal probability） 或*边际分布*（marginal distribution）：B指定一个数，横轴A作为随机变量，纵轴AB同时发生的概率
$$
P(B) = \sum_{A} P(A, B),
$$
根据边际分布求解连续随机变量的概率 

![image-20220826094614840](picture/image-20220826094614840.png)

两个变量是独立的时，P(A∣B)=P(A)。P(A,B)=P(A)P(B)。 P(A,B∣C)=P(A∣C)P(B∣C)，表示为A⊥B∣C。



期望表示随机变量的中心位置，如掷色子的期望值为3.5，扔硬币的期望值为0.5。
$$
E[X] = \sum_{x} x P(X = x).
$$
方差用于表示数据的分散程度。数据波动越大，方差就越大。

![image-20220826100419504](picture/image-20220826100419504.png)



# 线性神经网络



### 手动实现线性回归

```python
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 线性回归模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

# 均方损失
def squared_loss(y_hat, y): 
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 小批量随机梯度下降
def sgd(params, lr, batch_size):  
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

```python
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```



### 线性回归实现

```python
from d2l import torch as d2l
from torch.utils import data
import numpy as np
import torch
from torch import nn

# 构造数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 自定义数据迭代器，每次从数据集中不重复的拿出10条数据，直至拿完所有数据
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
data_iter = load_array((features, labels), 10)  # data_iter 是一个迭代器，是一个方法

# nn是神经网络的缩写
net = nn.Sequential(nn.Linear(2, 1))

# 初始化参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化方法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 拿出30条数据进行 3轮训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad() # 梯度清零
        l.backward() #反向传播
        trainer.step() # 模型更新
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 模型评价
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
```



### Softmax回归

```

```



```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# 两层网络，nn.Flatten用于调整输入数据的形状。没有softmax层，因为损失函数中已经实现了
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

# 只对模型中的线性模型层进行初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# 损失函数
loss = nn.CrossEntropyLoss(reduction='none')





# 训练
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```



# 多层感知机

*多层感知机*（multilayer perceptron），通常缩写为*MLP*：许多全连接层堆叠在一起



### 激活函数

$$
\operatorname{ReLU}(x) = \max(x, 0)
$$

使用ReLU的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过。 这使得优化表现得更好，并且ReLU减轻了困扰以往神经网络的梯度消失问题

```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```


$$
\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}
$$

```python
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

x.grad.data.zero_()   # 清除以前的梯度
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```


$$
\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}
$$

```python
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```



### 多层感知机实现

 感知机不能解决XOR问题，即异或问题。

```python
import torch
from torch import nn
from d2l import torch as d2l

# 定义神经网络
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

# 初始化参数权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

# 定义损失函数和优化函数
loss = nn.CrossEntropyLoss(reduction='none')
lr= 0.1
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# 批量读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 模型训练
if isinstance(net, torch.nn.Module): # 判断模型net是否符合要求
    net.train() # 开启模型训练模式，https://zhuanlan.zhihu.com/p/458332467

num_epochs = 10
for epoch in range(num_epochs):
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(trainer, torch.optim.Optimizer): # 下面的训练过程没看懂
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
        else:
            l.sum().backward()
            trainer(X.shape[0])
```



### 一些理论

 独立同分布假设：我们假设训练数据和测试数据都是从相同的分布中独立提取的。 

欠拟合：训练误差和验证误差都很严重， 但它们之间*泛化误差*很小。需要采用更复杂的模型。

过拟合：训练误差明显低于验证误差。

我们通常更关心验证误差，而不是训练误差和验证误差之间的差距。



### 正则化

 将权重衰减集成到优化算法中

```python
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    
    for param in net.parameters():
        param.data.normal_()
        
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
    print('w的L2范数：', net[0].weight.norm().item())
```



### Dropout

dropout ，在隐藏层随机丢弃神经元，起到正则化的效果。

在训练阶段使用，预测阶段不使用。

```python
dropout1, dropout2 = 0.2, 0.5

net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```



### 前向传播

前向传播计算图。

下图有的是函数，有的是变量。

![../_images/forward.svg](picture/forward.svg)

z是线性函数，h是激活函数，o是下一个线性函数，l是损失函数，s是正则函数，j是目标函数。



### 反向传播

https://zhuanlan.zhihu.com/p/40378224     就是反向链式求导。



反向传播比正向传播的优势：

1. 可以存储计算结果复用。
2. 因为输出维度远小于输入维度，所以反向传播可以减少计算量。



### 梯度传递

使用梯度下降对神经网络进行层层求梯度时，无论正向传播还是反向传播，梯度层层传递，就会发生**梯度爆炸**或者**梯度消失**。

现在一般是使用BP（反向传播算法）求解神经网络，会用到链式法则，使梯度连乘。



影响产生梯度爆炸或者梯度消失的几个原因：

1.深层网络，网络越深，梯度指数变化。梯度消失的时候底部层跑不动。

2.激活函数，不同激活函数，求导后对梯度有影响。

3.初始权重，初始权重过大，梯度爆炸。



解决方法:

1.梯度剪切，强制将梯度限制在一定范围内。

2.加入正则，防止梯度爆炸，因为梯度爆炸时w也会很大，正则控制了w。

3.合理的初始权重和激活函数。

4.梯度归一化，对每一层的输出规范为均值和方差一致。

5.残差网络的捷径（shortcut）。

6.LSTM的“门（gate）”结构。



### kaggle实战

https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/kaggle-house-price.html



# 深度学习

神经网络为什么要深度？

每层都提取到了有效信息，进行组合，节省了空间。如果直接在一层网络组合，会有很多无效组合。



### 层与块

块：比单个层大”但“比整个模型小”的组件



自定义块

```python
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```

块的一个主要优点是它的多功能性。 我们可以子类化块以创建层（如全连接层的类）、 整个模型（如上面的`MLP`类）或具有中等复杂度的各种组件。

块可以用 Sequential 连接起来。可以通过块和Sequential 嵌套任意组合网络。



自定义层

```python
import torch
import torch.nn.functional as F
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
  
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

# 带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
   
linear = MyLinear(5, 3)
linear.weight
```



### 参数管理

参数获取：参数是对象，包含值、梯度和其他信息。

```python
# 查所有
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

# 查
print(net[2].state_dict())
print(net[2].bias)
print(net[2].bias.data)
print(rgnet[0][1][0].bias.data)
print(net[2].weight.grad)
```

参数初始化

```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]

# 对不同层采用不同的初始化方法
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]

#直接修改参数
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

参数绑定

```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), 
                    nn.ReLU(),
                    shared,
                    nn.ReLU(),
                    shared,
                    nn.ReLU(),
                    nn.Linear(8, 1))
net.apply(init_normal)
print(net[2].weight.data[0] == net[4].weight.data[0])
```



### 保存 & 加载

保存加载张量

```python
import torch
from torch import nn
from torch.nn import functional as F

# 保存张量
x = torch.arange(4)
torch.save(x, 'x-file')

# 加载张量
x2 = torch.load('x-file')

# 保存加载多个张量
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')

# 保存加载多个张量
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
```

保存加载参数（模型需要提前定义好相同模型架构，无法随参数一起保存加载）

```python
# 保存参数
torch.save(net.state_dict(), 'mlp.params')

# 加载参数
net.load_state_dict(torch.load('mlp.params'))
clone.eval()
```



### 使用GPU

```python
import torch
from torch import nn

# 查看GPU数量
torch.cuda.device_count()

# 查询所有gpu，否则返回cpu
def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

# 查询指定gpu，否则返回cpu
def try_gpu(i=0): 
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
```

```python
# 为张量指定存储device
X = torch.ones(2, 3, device=try_gpu())

# 复制GPU0的张量X到GPU1的张量Z，只有同一device的数据才能进行计算。
Z = X.cuda(1)

# 为神经网络参数指定device
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
net[0].weight.data.device #查看
```



使用多GPU 或者 转换为numpy 都会使数据在多个device间传输，相比于计算，数据复制很慢。

所以尽量减少数据传输，让操作在一个位置完成。



# CNN

*卷积神经网络*（convolutional neural network，CNN）

基于卷积神经网络架构的模型在计算机视觉领域中已占主导地位，当今几乎所有的图像识别、目标检测或语义分割相关的学术竞赛和商业应用都以此为基础。



### 基本概念

https://zhuanlan.zhihu.com/p/42559190

卷积：核和图像得一部分对应元素相乘相加得到输出。

滤波器（filter，也称为kernel）：跟图像中的某块区域做运算的矩阵。

步长：核每次移动的距离。

padding：卷积使得图像缩小，卷积对原图的边缘学习不够，使用扩大原图的方法避免这两个问题。

池化：核与图像做的运算。 



### CNN优点

CNN对比全连接神经网络：

CNN，无非就是把FC（全连接层）改成了CONV（卷积层）和POOL（池化层），就是把传统的由一个个神经元组成的layer，变成了由filters组成的layer。

参数减少，全连接层参数数量由输入特征和神经元个数决定，卷积层参数由核的大小决定。因为一个卷积层只有一个核和不同输入做运算。

卷积操作可以使得相同得输入经过同一核得到相同得输出，这是平移不变性。

输出只和输入得一部分有关，更好得学习了图中得一些结构，这是局部性。



适合于计算机视觉的神经网络架构：

1. *平移不变性*（translation invariance）：不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应。
2. *局部性*（locality）：神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系。最终，可以聚合这些局部特征，以在整个图像级别进行预测。



### 多通道？

通道：图像中每个像素由RGB3个通道，计算中就是多了一个维度。下面待整理。

多输入通道：当输入包含多个通道时，需要构造一个与输入数据具有相同输入通道数的卷积核，以便与输入数据进行互相关运算。

多输出通道：在最流行的神经网络架构中，随着神经网络层数的加深，我们常会增加输出通道的维数，通过减少空间分辨率以获得更大的通道深度？

```python
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

# 卷积核
K = torch.stack((K, K + 1, K + 2), 0)
```



1×1卷积：kernel 大小是1×1，在高度和宽度维度上失去了卷积层的识别相邻元素间相互作用的能力。 其实1×1卷积的唯一计算发生在通道上。



### 池化层

池化：核与输入不做卷积运算，而是取最大，取平均

优点：降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性。

多通道：在处理多通道输入数据时，池化层在每个输入通道上单独运算。 这意味着池化层的输出通道数与输入通道数相同



### LeNet

LeNet，最早发布的卷积神经网络之一。



总体来看，LeNet（LeNet-5）由两个部分组成：

- 卷积编码器：由两个卷积层组成;
- 全连接层密集块：由三个全连接层组成。

![../_images/lenet.svg](picture/lenet.svg)

每个卷积块中的基本单元是一个卷积层、一个sigmoid激活函数和平均汇聚层。请注意，虽然ReLU和最大汇聚层更有效，但它们在20世纪90年代还没有出现。

每个卷积层使用5×5卷积核和一个sigmoid激活函数。这些层将输入映射到多个二维特征输出，

通常同时增加通道的数量。第一卷积层有6个输出通道，而第二个卷积层有16个输出通道。

每个2×2池操作（步幅2）通过空间下采样将维数减少4倍。

卷积的输出形状由批量大小、通道数、高度、宽度决定。

LeNet的稠密块有三个全连接层，分别有120、84和10个输出。

```python
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

# 查看输出
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
```

模型训练

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 模型评价
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval() 
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X] # BERT微调所需的（之后将介绍）
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 模型训练
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    # 参数初始化
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    # 指定GPU
    print('training on', device)
    net.to(device)
    # 损失函数核优化函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)
    # 模型多轮训练
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')
    
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu()) 
```



# 现代卷积神经网络

### AlexNet

从LeNet（左）到AlexNet（右）

![../_images/alexnet.svg](picture/alexnet.svg)

AlexNet使用 ReLU 而不是 sigmoid 作为其激活函数。

AlexNet通过暂退法控制全连接层的模型复杂度，而LeNet只使用了权重衰减。

```python
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    # 这里，我们使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))

# 查看网络结构
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
    
# 读取数据
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

# 模型训练
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

Dropout、ReLU和预处理是提升计算机视觉任务性能他关键步骤。



### VGG

一个VGG块，由一系列卷积层组成，再加上用于空间下采样的最大汇聚层。

```python
import torch
from torch import nn
from d2l import torch as d2l

# 卷积层的数量num_convs、输入通道的数量in_channels 和输出通道的数量out_channels.
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```



VGG网络可以分为两部分：第一部分主要由卷积层和汇聚层组成，第二部分由全连接层组成。与AlexNet、LeNet一样。

![../_images/vgg.svg](picture/vgg.svg)

```python
# VGG-11网络
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
        
    return nn.Sequential(
        *conv_blks, 
        nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)

# 查看网络形状
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```



### NiN

卷积层的输入和输出由四维张量组成，张量的每个轴分别对应样本、通道、高度和宽度。

全连接层的输入和输出通常是分别对应于样本和特征的二维张量。

NiN是在每个像素位置（针对每个高度和宽度）应用一个全连接层。即将空间维度中的每个像素视为单个样本，将通道维度视为不同特征（feature）。

NiN块以一个普通卷积层开始，后面是两个1×1的卷积层。这两个1×1卷积层充当带有ReLU激活函数的逐像素全连接层。 

第一层的卷积窗口形状通常由用户设置。 随后的卷积窗口形状固定为1×1。

![../_images/nin.svg](picture/nin.svg)

```python
import torch
from torch import nn

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
```

```python
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())
```



### GoogLeNet

Inception块

![../_images/inception.svg](picture/inception.svg)

```python
import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)
```



GoogLeNet

![../_images/inception-full.svg](picture/inception-full.svg)

```python
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```



### batch normalization

批量规范化解决的问题：https://zhuanlan.zhihu.com/p/52736691

1.缓解了梯度传递问题。

2.起到了正则化作用



批量规范化：在每次训练迭代中，从批量的X中得到均值与方差，对批量的X进行规范化。

`在应用批量规范化时，批量大小的选择可能比没有批量规范化时更重要`
$$
\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.
$$
输入x减去均值除以方差，我们经常会对x进行标准化处理，使得均值为0，标准差为1。γ 和 β 是对 x 进行拉伸和偏移，也是需要学习的参数。



应用于单个可选层（也可以应用到所有层）,可持续加速深层网络的收敛速度。



批量规范化层在”训练模式“（通过小批量统计数据规范化）和“预测模式”（通过数据集统计规范化）中的功能不同。小批量和全部的区别。



在全连接层和卷积层位置不同：

在全连接层中 是wx+b之后，激活函数之前。

在卷积层是卷积操作之后，激活函数之前，所有x共用均值和方差，但是γ 和 β每个通道都不相同。



手动实现

```python
import torch
from torch import nn
from d2l import torch as d2l

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data
```



继承神经网络模型类进行手动实现

```python
class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```



LeNet + BatchNorm pytorch实现

```python
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))

d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```



### 残差网络(ResNet)

更复杂的嵌套函数更容易包含最优解，但是更复杂的非嵌套函数不一定跟接近最优解，在此基础上提出了残差网络。

![../_images/functionclasses.svg](picture/functionclasses.svg)

 残差网络的核心思想是：每个附加层都应该更容易地包含原始函数作为其元素之一。



ResNet沿用了VGG完整的3×3卷积层设计。

**残差块**里首先有2个有相同输出通道数的3×3卷积层。 每个卷积层后接一个批量规范化层和ReLU激活函数。

 然后我们通过跨层数据通路，跳过这2个卷积运算，将输入直接加在最后的ReLU激活函数前。 

这样的设计要求2个卷积层的输出与输入形状一样。 如果想改变通道数，就需要引入一个额外的1×1卷积层来将输入变换成需要的形状后再做相加运算。

![../_images/resnet-block.svg](picture/resnet-block.svg)

残差块实现

```python
import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    
X = torch.rand(4, 3, 6, 6)

# use_1x1conv=False，输入和输出形状一致
blk = Residual(3,3)
blk(X).shape

#use_1x1conv=True，输入和输出形状不一致
blk = Residual(3,6, use_1x1conv=True, strides=2)
blk(X).shape
```



ResNet-18

![../_images/resnet18.svg](picture/resnet18.svg)

ResNet 实现

```python
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                   
def resnet_block(input_channels, num_channels, num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
```



### DenseNet

稠密连接示意图

![../_images/densenet.svg](picture/densenet.svg)



稠密网络主要由2部分构成：*稠密块*（dense block）和*过渡层*（transition layer）



稠密块体

一个*稠密块*由多个卷积块组成，每个卷积块使用相同数量的输出通道。 在前向传播中，我们将每个卷积块的输入和输出在通道维上连结。

```python
import torch
from torch import nn

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X
```



https://zh-v2.d2l.ai/chapter_convolutional-modern/densenet.html





































































