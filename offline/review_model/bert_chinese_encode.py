import torch
from transformers import BertTokenizer, BertModel

# 加载字符映射器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 加载预训练中文模型
model = BertModel.from_pretrained('bert-base-chinese')


def get_bert_encode_for_single(text):
    """
    功能: 使用bert-chinese预训练模型对中文文本进行编码
    text: 要进行编码的中文文本
    return : 编码后的张量
    """

    # 首先使用字符映射器对每个汉子进行映射
    # bert中的tokenizer映射后会加入开始和结束的标记, 101, 102, 这两个标记对我们不需要，采用切片的方式去除
    indexed_tokens = tokenizer.encode(text)[1:-1]

    # 封装成tensor张量
    tokens_tensor = torch.tensor([indexed_tokens])
    # print(tokens_tensor)

    # 预测部分需要使得模型不自动求导
    with torch.no_grad():
        last_hidden_state = model(tokens_tensor)[0]

    # print(encoded_layers.shape)
    # 模型的输出都是三维张量,第一维是1,使用[0]来进行降维,只提取我们需要的后两个维度的张量
    return last_hidden_state[0]


if __name__ == '__main__':
    text = "你好"
    outputs = get_bert_encode_for_single(text)
    print(outputs)
    print(outputs.shape)
