https://www.youtube.com/watch?v=JGtqpQXfJis&ab_channel=Hung-yiLee



# 概要

生成式人工智能：输出是**复杂**且**有结构**的人工智能。复杂 指复杂到不能穷举。



我可以做什么：多种方法使用大模型，训练自己的模型



# 使用大模型

不训练大模型，又增强大模型的方法，本质上都是增加了输入或者输出，因为自回归，增加的 输入或者输出 又增强了输出。



### Prompt 工程

主旨：不用写特定格式的 Prompt ，交代清楚全面即可。



------

**咒语**（在特定的模型上可能起效）：



类型1：Chain of Thought (CoT)

Let's think step by step。
Answer by starting with "Analysis:" 



类型2：情绪勒索

This is very important to my career.

做的好有小费，做的不好有处罚



类型3：让模型给咒语



### in-context learning

给与模型更多的上下文，提供更多的信息。



方式包括：

1、提问中，给出更多的前提信息。

2、通过 文件、网络 获得更多的上下文

3、提供任务范例，让模型学习。



### 拆解任务



1、拆解：将一个复杂的任务拆解成多个步骤，一个步骤一个步骤的完成

2、检查：直接要求检查答案，或者把问答作为输入，进行检查。

3、Self-Consistency：多次答案取众数

4、综合方案

![image-20250918152232537](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20250918152232537.png)



### 使用工具

RAG（搜索引擎，向量数据库）、写代码（POT），文生图（DALL-E）



怎么使用工具，本质还是文字接龙，接龙就有可能选错工具

![image-20250918204918708](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20250918204918708.png)



### 模型合作 

1、决策模型决定使用其他那些模型，协助完成工作。

2、模型互相讨论（给出要反对对方的提示词，有裁判模型判定讨论结束）

3、使用不同模型，扮演不同角色，团队合作。包括考核角色，组织角色，工作角色



# 训练大模型



### 背景知识

语言大模型：文字接龙 模型，一个个产生 token



需要多少文字训练：非常多

语言知识

世界知识







Self-supervised Learning（自监督学习)

过滤有害内容，去除无意义符号，去除低品质资料，去除重复资料































































