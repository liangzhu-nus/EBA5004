# AI_liitle_p
Dialogue system fine-tuned based on BERT model, focusing mainly on disease symptoms

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