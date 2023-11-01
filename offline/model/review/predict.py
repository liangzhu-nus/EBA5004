import os
import torch
import torch.nn as nn
from rnn_model import RNN
from bert_chinese_encode import get_bert_encode_for_single

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 从当前目录返回到上级目录（ner）
OFFLINE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))

# 设定预加载的模型路径
MODEL_PATH = f'{CURRENT_DIR}/BERT_RNN.pth'

# 设定若干参数, 注意：这些参数一定要和训练的时候保持完全一致!!!
n_hidden = 128
input_size = 768
n_categories = 2

# 实例化RNN模型，并加载保存的模型参数
rnn = RNN(input_size, n_hidden, n_categories)
rnn.load_state_dict(torch.load(MODEL_PATH))


# 编写测试函数
def _test(line_tensor):
    """用于调用RNN模型并返回结果
    Args:
        line_tensor (_type_): 代表输入中文文本的张量标识
    """
    # 初始化隐藏层
    hidden = rnn.initHidden()

    # 遍历输入文本中的每一个字符张量
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)

    # 返回RNN模型的最终输出
    return output


def predict(input_line):
    """预测函数
    Args:
        input_line (_type_): 代表需要预测的中文文本信息
    """
    # 注意: 所有的预测必须保证不自动求解梯度
    with torch.no_grad():
        # 将input_line使用bert模型进行编码，然后将张量传输给_test()函数
        output = _test(get_bert_encode_for_single(input_line))

        # 从output中取出最大值对应的索引，比较的维度是1
        _, topi = output.topk(1, 1)
        return topi.item()


# 编写批量预测的函数
def batch_predict(input_path, output_path):
    """批量预测函数

    Args:
        input_path (_type_): 以原始文本的输入路径(等待进行命名实体审核的文件)
        output_path (_type_): 预测后的输出文件路径(经过命名实体审核通过的所有数据)
    """
    csv_list = os.listdir(input_path)

    # 遍历每一个csv文件
    for csv in csv_list:
        # 要以读的方式打开每一个csv文件
        with open(os.path.join(input_path, csv), "r") as fr:
            # 要以写的方式打开输出路径下的同名csv文件
            with open(os.path.join(output_path, csv), "w") as fw:
                # 读取csv文件的每一行
                input_line = fr.readline()
                # 调用预测函数，利用RNN模型进行审核
                res = predict(input_line)
                if res:
                    # 如果res==1, 说明通过了审核
                    fw.write(input_line + "\n")
                else:
                    pass


if __name__ == '__main__':
    input_path = f"{OFFLINE_DIR}/datasets/structured/noreview/"
    output_path = f"{OFFLINE_DIR}/datasets/structured/reviewed/"
    batch_predict(input_path, output_path)
