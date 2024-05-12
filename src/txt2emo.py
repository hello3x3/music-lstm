import torch
from torch import nn
from transformers import BertModel
from transformers import BertTokenizer


def get_label_string(label):
    labels = {
        "喜悦": 0,
        "愤怒": 1,
        "厌恶": 2,
        "低落": 3
    }
    for key, value in labels.items():
        if value == label:
            return key
    return None


# 构建模型
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('./bert',num_labels=4)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


model = BertClassifier()
model.load_state_dict(torch.load('BERT-weibo.pt'))
model.eval()
tokenizer = BertTokenizer.from_pretrained('./bert')

# 输入测试文本
text = input()
text_input = tokenizer(text,padding='max_length',max_length = 16,truncation=True,return_tensors="pt")
mask = text_input['attention_mask']
input_id = text_input['input_ids']
output = model(input_id, mask)
output = output.argmax(dim=1)
output = output.item()
label_string = get_label_string(output)

print("---------测试文本----------")
print(text)
print("---------结果----------")
print(label_string)
