import os
from src.config import *
from torch import nn
from transformers import BertModel


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes) -> None:
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model())
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)


# net = SentimentClassifier(3)
# print(net)