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
        self.hidden_size = hidden_size // 2  # 设置隐藏层维度, 若为双向时想要得到同样大小的向量, 需要除以2
        self.batch_size = batch_size  # 设置批次大小, 对应每个批次的样本条数
        self.sentence_lenth = sentence_length  # 设定句子长度
        self.batch_first = batch_first
        self.num_layers = num_layers  # 设置网络的LSTM层数
        
        # 构建词嵌入层: 字向量, 维度为总单词数量与词嵌入维度
        # 构建双向LSTM层
        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.bilstm = nn.LSTM(input_size = input_feature_size,
                              hidden_size = self.hidden_size,
                              num_layers = num_layers,
                              bidirectional = True,
                              batch_first = batch_first)
        
        # 构建全连接线性层: 将BiLSTM的输出层进行线性变换
        self.linear = nn.Linear(hidden_size, self.tag_size)
        
# 参数1:码表与id对照
char_to_id = {"双": 0, "肺": 1, "见": 2, "多": 3, "发": 4, "斑": 5, "片": 6,
              "状": 7, "稍": 8, "高": 9, "密": 10, "度": 11, "影": 12, "。": 13}

# 参数2:标签码表对照
tag_to_id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}

# 参数3:字向量维度
EMBEDDING_DIM = 200

# 参数4:隐层维度
HIDDEN_DIM = 100

# 参数5:批次大小
BATCH_SIZE = 8

# 参数6:句子长度
SENTENCE_LENGTH = 20

# 参数7:堆叠 LSTM 层数
NUM_LAYERS = 1

# 初始化模型
model = BiLSTM(vocab_size=len(char_to_id),
               tag_to_id=tag_to_id,
               input_feature_size=EMBEDDING_DIM,
               hidden_size=HIDDEN_DIM,
               batch_size=BATCH_SIZE,
               sentence_length=SENTENCE_LENGTH,
               num_layers=NUM_LAYERS)
print(model)

# if __name__ == '__main__':
    