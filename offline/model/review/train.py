from bert_chinese_encode import get_bert_encode_for_single
import torch
import torch.nn as nn
import random
import math
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
from rnn_model import RNN


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# 读取数据
train_data_path = f'{CURRENT_DIR}/train_data.csv'
train_data = pd.read_csv(train_data_path, header=None, sep='\t')

# 打印一下正负标签比例
# print(dict(Counter(train_data[0].values)))

# 打印若干数据展示一下
train_data = train_data.values.tolist()
# print(train_data1[:10])


def randomTrainingExample(train_data):
    """随机选取数据函数, train_data是训练集的列表形式数据"""
    # 从train_data随机选择一条数据
    category, line = random.choice(train_data)
    # 将里面的文字使用bert进行编码, 获取编码后的tensor类型数据
    line_tensor = get_bert_encode_for_single(line)
    # 将分类标签封装成tensor
    category_tensor = torch.tensor([int(category)])
    # 返回四个结果
    return category, line, category_tensor, line_tensor


# 选取损失函数为nn.NLLLoss()
criterion = nn.NLLLoss()

hidden_size = 128
# 预训练模型bert输出的维度
input_size = 768
n_categories = 2
rnn = RNN(input_size, hidden_size, n_categories)

# 把学习率设定为0.005
learning_rate = 0.005


def train(category_tensor, line_tensor):
    # category_tensor: 代表类别的张量, line_tensor: 代表经过bert编码后的文本张量
    # 初始化隐藏层
    hidden = rnn.initHidden()

    # 训练前一定要将梯度归零
    rnn.zero_grad()

    # 遍历line_tensor中的每一个字符的张量
    for i in range(line_tensor.size()[0]):
        # 传入rnn中的参数必须是二维张量,如果不是,需要扩展维度 unsqueeze(0)
        output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)

    # 调用损失函数, 输入分别是rnn预测的结果和真实的类别标签
    loss = criterion(output, category_tensor)

    # 开启反向传播
    loss.backward()

    # 为大家显示的更新模型中的所有参数
    for p in rnn.parameters():
        # 利用梯度下降法更新, add_()功能是参数的梯度乘以学习率，然后结果相加来更新参数
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


def valid(category_tensor, line_tensor):
    # category_tensor: 类别标签的张量, line_tensor: 经过了bert编码后的文本张量
    # 初始化隐藏层
    hidden = rnn.initHidden()

    # 注意: 验证函数中要保证模型不自动求导
    with torch.no_grad():
        # 遍历文本张量中的每一个字符的bert编码
        for i in range(line_tensor.size()[0]):
            # 注意: 输入rnn的参数必须是二维张量,如果不足,利用unsqueeze()来进行扩展
            output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)

        loss = criterion(output, category_tensor)

    return output, loss.item()


def timeSince(since):
    # 功能:获取每次打印的时间消耗, since是训练开始的时间
    # 获取当前的时间
    now = time.time()

    # 获取时间差, 就是时间消耗
    s = now - since

    # 获取时间差的分钟数
    m = math.floor(s/60)

    # 获取时间差的秒数
    s -= m*60

    return '%dm %ds' % (m, s)


# 设置训练的迭代次数
n_iters = 5000

# 设置打印间隔为100
plot_every = 100

# 初始化训练和验证的损失，准确率
train_current_loss = 0
train_current_acc = 0
valid_current_loss = 0
valid_current_acc = 0

# 为后续的画图做准备，存储每次打印间隔之间的平均损失和平均准确率
all_train_loss = []
all_train_acc = []
all_valid_loss = []
all_valid_acc = []

# 获取整个训练的开始时间
start = time.time()

# 进入主循环,遍历n_iters次
for iter in range(1, n_iters + 1):
    # 分别调用两次随机获取数据的函数，分别获取训练数据和验证数据
    category, line, category_tensor, line_tensor = randomTrainingExample(train_data)
    category_, line_, category_tensor_, line_tensor_ = randomTrainingExample(train_data)

    # 分别调用训练函数，和验证函数，得到输出和损失
    train_output, train_loss = train(category_tensor, line_tensor)
    valid_output, valid_loss = valid(category_tensor_, line_tensor_)

    # 累加训练的损失，训练的准确率，验证的损失，验证的准确率
    train_current_loss += train_loss
    train_current_acc += (train_output.argmax(1) == category_tensor).sum().item()
    valid_current_loss += valid_loss
    valid_current_acc += (valid_output.argmax(1) == category_tensor_).sum().item()

    # 每隔plot_every次数打印一下信息
    if iter % plot_every == 0:
        train_average_loss = train_current_loss / plot_every
        train_average_acc = train_current_acc / plot_every
        valid_average_loss = valid_current_loss / plot_every
        valid_average_acc = valid_current_acc / plot_every

        # 打印迭代次数，时间消耗，训练损失，训练准确率，验证损失，验证准确率
        print("Iter:", iter, "|", "TimeSince:", timeSince(start))
        print("Train Loss:", train_average_loss, "|", "Train Acc:", train_average_acc)
        print("Valid Loss:", valid_average_loss, "|", "Valid Acc:", valid_average_acc)

        # 将损失，准确率的结果保存起来，为后续的画图使用
        all_train_loss.append(train_average_loss)
        all_train_acc.append(train_average_acc)
        all_valid_loss.append(valid_average_loss)
        all_valid_acc.append(valid_average_acc)

        # 将每次打印间隔的训练损失，准确率，验证损失，准确率，归零操作
        train_current_loss = 0
        train_current_acc = 0
        valid_current_loss = 0
        valid_current_acc = 0


plt.figure(0)
plt.plot(all_train_loss, label="Train Loss")
plt.plot(all_valid_loss, color="red", label="Valid Loss")
plt.legend(loc="upper left")
plt.savefig(f"{CURRENT_DIR}/loss.png")

plt.figure(1)
plt.plot(all_train_acc, label="Train Acc")
plt.plot(all_valid_acc, color="red", label="Valid Acc")
plt.legend(loc="upper left")
plt.savefig(f"{CURRENT_DIR}/acc.png")


# 模型的保存，首先给定保存的路径
MODEL_PATH = f'{CURRENT_DIR}/BERT_RNN.pth'

torch.save(rnn.state_dict(), MODEL_PATH)
