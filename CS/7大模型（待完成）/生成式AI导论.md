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

语言大模型：本质是 **文字接龙** 模型，一个个产生 token。



需要多少文字训练：非常多

需要学到 **语言知识、世界知识**。



模型训练分为两个阶段，第一阶段，叫 **foundation model** ，第二阶段叫 **Alignment** 对齐。

对于语言大模型来说，第一阶段包括 **Self-supervised Learning** ，第二阶段包括 **Instruction Fine-tuning** 和 **RLHF**。



### Self-supervised Learning

自监督学习：无需人类打标签进行学习。大语言模型本身的训练就是自监督学习。



过程：

1、从网络上搜集大量资料。

2、经过数据处理：过滤有害内容，去除无意义符号，去除低品质资料，去除重复资料。

3、训练出基础的大语言模型。



大语言模型的训练成本非常高，只有超级大公司有资本训练。

其他人获取到 foundation model 的方法，利用开源大模型的 开放出的参数，LLaMA 第一个开源了 大模型 pre-train的参数。



### Instruction Fine-tuning

微调：利用少量的资料，来调整模型，使得模型更准确或者更专业的过程。是画龙点睛的作用。

微调是 **supervised Learning**，需要靠大量人类产生资料，进行资料标注。



微调需要在 foundation model 的基础上。对于微调来说，之前训练基础模型的过程叫做 pre-train。

微调复用 pre-train 参数的方法叫做 **Adapter**，Adapter中最常用的方法是 **LoRA**。

![image-20250920181046245](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20250920181046245.png)



Fine-tuning 两条路

1、专才：代表模型 BERT ，打造一堆专才，针对基础模型只微调一个功能，比如翻译，问答等。

2、通才：打造通才。用所有任务去微调。



微调 任务资料 获取：

1、人类直接标注：把之前的 NLP 任务，标记成问答的形式。比较死板，经常与实际问答场景不符合。

2、网络收集+标记员：从大模型网站上收集到人们的真实问题，标记员来回答。

3、**逆向工程 ChatGPT** ：直接让  ChatGPT 出任务，问题，回答。这是投入最小的方法。



### RLHF

Reinforcement Learning from Human Feedback



简单理解：人类反馈回答的两个答案，哪个好。然后微调，使得好答案几率高。



**RLAIF**：使用人类反馈的答案，训练 **回馈模型 Reward model**，让回馈模型判断模型的哪个答案好，然后做RL。

注意，过度使用回馈模型，会使效果下降。



RLHF 的局限性：

1、好的标准多样，比如更 安全，还是更 有帮助。

2、有些复杂问题，人类都不知道哪个好。



# AI Agent

典型的 Agent 如：**AutoGPT**

![image-20250920214623745](https://raw.githubusercontent.com/zhanghongyang42/images/main/image-20250920214623745.png)



# 模型可解释性

可以从多个方面找到模型的可解释性。



一、**找出影响输出的关键输入**

1、随机去除输入的一个token，看对输出的影响。

2、看输入输出的gradient

3、看 attention 的weight



二、**找到影响输出的关键资料**



三、**找到embedding 中的信息**



四、**测谎器模型**

用真话假话，训练一个测谎器模型，让测谎器模型测试大模型。



五、**直接问大语言模型，但可能不对**



# 检测模型能力

使用标准答案 **Benchmark** 检验模型的能力。网络上 有很多的标准的 Benchmark 。

除了检验模型效果，还可以从速度，价格方面考察大模型。



两种和Benchmark 比较的方法：

1、同样的输入，给不同的模型，把不同模型的输出，和标准答案 Benchmark 比较。

2、直接把一个模型的输出和 Benchmark 比较。



使用 Benchmark 的方法：

1、传统方法：一些规则方法。但是因为语言模型的输出是不可控的，传统方法很难准确的确定模型的输出是否和 Benchmark 一致，不是100%评测准确。

2、人类评估

3、大语言模型：代替人类，进行评测。模型评测可能不准确，比如偏向长输出，可以使用其他方法校正一部分。



注意：Benchmark 的考题因为是公开的，所以可能被模型学习到，不可尽信。













