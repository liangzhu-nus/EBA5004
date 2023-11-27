# AI_liitle_p
**项目简介：**
根据用户输入症状来匹配对应疾病.
Dialogue system fine-tuned based on BERT model, focusing mainly on disease symptoms


## 命名实体识别

**What：**
整体流程：将文本信息(文本数字编码化)经过词嵌入层, BiLSTM层, 线性层的处理, 最终输出句子的张量。张量的最后一维是每一个word映射到7个标签的概率, 发射矩阵。
使用维特比算法通过转移矩阵和发射矩阵得出最可能的序列，发射矩阵作为模型参数的一部分，通过模型训练得出。

本质：序列标注问题：使用BiLSTM+CRF模型。输出句子中每个字的标签。自定义标签：{"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, START_TAG: 5, STOP_TAG: 6}。
再转换为 以disease 为文件名，内容为症状的文件。

训练代码：offline/model/ner/train.py
训练数据：offline/model/ner/data/train.txt
1. 将训练数据集转换为数字化编码集(根据中文字符向id的映射表)，生成了新的数据集文件 train.npz。
2. 字嵌入或词嵌入作为BiLSTM+CRF模型的输入, 而输出的是句子中每个单元的标签.
   2.1 BiLSTM层的输出为每一个标签的预测分值（发射矩阵）。标签是：tag_to_ix = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, START_TAG: 5, STOP_TAG: 6}
   2.2 CRF层可以为最后预测的标签添加一些约束来保证预测的标签是合法的. 在训练数据训练的过程中, 这些约束可以通过CRF层自动学习到.（输入发射矩阵，输出最可能的概率序列）


**输入输出：**
将 offline/datasets/unstructured/norecognized 文件中的文本数据进行命名实体识别。



## 命名实体审核
**What:**
审核文件中的症状，将不合法的症状剔除掉。
文件说明：
noreview 文件夹下，每一个文件名对应一种疾病名，文件中内容表示对应的症状。

**输入输出**
将 offline/structured/noreview 中的数据通过命名实体审核提取到 offline/structured/reviewed 文件夹下，并将审核后的数据写入到 neo4j 中。

**How：**
本质：短文本二分类问题.
训练代码：offline/model/review/train.py
训练数据：offline/model/review/train_data.py
```
1	手掌软硬度异常
0	常异度硬软掌手
1	多发性针尖样瘀点
...
```
1. 使用了bert-chinese预训练模型获取 embedding 表示。
2. 使用 RNN 来做二分类。
3. 如果是同一类。识别文本中提及的疾病症状，并返回与这些症状相关联的疾病名称列表。



## 两个句子相关性
数据集：online/bert_server/datasets/train_data.csv
编码：使用 bert 对原始文本进行编码
https://huggingface.co/docs/transformers/model_doc/bert
步骤：
1. 文本数据编码-feature encoding
2. 下游分类任务模型
3. 构建数据加载器函数.
4. 构建模型训练函数.
5. 构建模型验证函数.
6. 调用训练和验证函数并打印日志.
7. 绘制训练和验证的损失和准确率对照曲线.
8. 模型保存.