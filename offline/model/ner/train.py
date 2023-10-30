import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, tag_to_id, input_feature_size, hidden_size, batch_size, sentence_length, num_layers=1, batch_first=True):
        """_summary_

        Args:
            vocab_size (_type_): 所有句子包含字符大小
            tag_to_id (_type_): 标签与 id 对照
            input_feature_size (_type_): LSTM 输入层维度（字嵌入维度）
            hidden_size (_type_): 隐藏层向量维度
            batch_size (_type_): 批训练大小
            sentence_length (_type_): 句子长度
            num_layers (int, optional): LSTM 层数
            batch_first (bool, optional): 是否将 batch_size 放置到矩阵的第一维度
        """
        super().__init__()
        self.tag_to_id = tag_to_id
        self.tag_size = len(tag_to_id)
        self.embedding_size = input_feature_size
        self.hidden_size = hidden_size // 2 # 设置隐藏层维度, 若为双向时想要得到同样大小的向量, 需要除以2
        self.batch_size = batch_size # 设置批次大小, 对应每个批次的样本条数
        self.sentence_lenth = sentence_length # 设定句子长度
        self.batch_first = batch_first
        self.num_layers = num_layers # 设置网络的LSTM层数
        
        #  构建词嵌入层: 字向量, 维度为总单词数量与词嵌入维度
        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.bilstm = nn.LSTM(input_size = input_feature_size,
                              hidden_size = self.hidden_size,
                              num_layers = num_layers,
                              bidirectional = True,
                              batch_first = batch_first)
        
        # 构建全连接线性层: 将BiLSTM的输出层进行线性变换
        self.linear = nn.Linear(hidden_size, self.tag_size)
        
def sentence_map(sentence_list, char_to_id, max_length):
    """中文文本的数字化编码:

    Args:
        sentence_list (_type_): _description_
        char_to_id (_type_): _description_
        max_length (_type_): _description_
    """
    sentence_list.sort(key=lambda x: len(x), reverse=True)
    sentence_map_list = []
    for

# if __name__ == '__main__':
    