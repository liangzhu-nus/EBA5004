import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')


def get_bert_encode(text_1, text_2, mark=102, max_len=10):
    """使用bert中文模型对输入的文本进行编码

    Args:
        text_1 (_type_): first sentence
        text_2 (_type_): second sentence
        mark (int, optional): 当输入两个文本的时候, 有中间的特殊分隔符. Defaults to 102.
        max_len (int, optional):限制的最大语句长度, 如果大于max_len, 进行截断处理, 如果小于max_len, 进行0填充的处理. Defaults to 10.

    Returns:
        _type_:输入文本的bert编码
    """
    # 第一步使用tokenizer进行两个文本的字映射
    indexed_tokens = tokenizer.encode(text_1, text_2)
    # 要对两个文本进行补齐, 或者截断的操作
    # 首先要找到分隔标记的位置
    k = indexed_tokens.index(mark)

    # 第二步处理第一句话, 第一句话是[:k]
    if len(indexed_tokens[:k]) >= max_len:
        # 长度大于max_len, 进行截断处理
        indexed_tokens_1 = indexed_tokens[:max_len]
    else:
        # 长度小于max_len, 需要对剩余的部分进行0填充
        indexed_tokens_1 = indexed_tokens[:k] + (max_len - len(indexed_tokens[:k])) * [0]

    # 第三步处理第二句话, 第二句话是[k:]
    if len(indexed_tokens[k:]) >= max_len:
        # 长度大于max_len, 进行截断处理
        indexed_tokens_2 = indexed_tokens[k:k+max_len]
    else:
        # 长度小于max_len, 需要对剩余的部分进行0填充
        indexed_tokens_2 = indexed_tokens[k:] + (max_len - len(indexed_tokens[k:])) * [0]

    # 将处理后的indexed_tokens_1和indexed_tokens_2进行相加合并
    indexed_tokens = indexed_tokens_1 + indexed_tokens_2

    # 需要一个额外的标志列表, 来告诉模型那部分是第一句话, 哪部分是第二句话
    # 利用0元素来表示第一句话, 利用1元素来表示第二句话
    # 注意: 两句话的长度都已经被我们规范成了max_len
    segments_ids = [0] * max_len + [1] * max_len

    # 将两个列表封装成张量
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])

    # 利用模型进行编码不求导
    with torch.no_grad():
        # 使用bert模型进行编码, 传入参数tokens_tensor和segments_tensor,
        # 最终得到模型的输出encoded_layers
        last_hidden_state = model(tokens_tensor, token_type_ids=segments_tensor).last_hidden_state
        # shape: (batch_size, sequence_length, hidden_size)
    return last_hidden_state


if __name__ == '__main__':
    text_1 = "人生该如何起头"
    text_2 = "改变要如何起手"

    last_hidden_state = get_bert_encode(text_1, text_2)
    print('encoded_layers', last_hidden_state)
    print('encoded_layers.shape', last_hidden_state.shape)
