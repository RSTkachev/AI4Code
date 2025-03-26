import torch
import torch.nn as nn

from transformers import BertModel


class OrderPredictionModel(nn.Module):
    def __init__(self, hidden_dim, dropout_prob=0.1):
        super(OrderPredictionModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.type_embedding = nn.Embedding(2, 8)
        self.fc1 = nn.Linear(768 * 2 + 8 * 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids1, att_mask1, cell_type1, input_ids2, att_mask2, cell_type2):
        with torch.no_grad():
            embedding1 = self.bert(input_ids1, attention_mask=att_mask1).pooler_output
            embedding2 = self.bert(input_ids2, attention_mask=att_mask2).pooler_output

        type_emb1 = self.type_embedding(cell_type1)
        type_emb2 = self.type_embedding(cell_type2)

        combined = torch.cat([embedding1, type_emb1, embedding2, type_emb2], dim=1)
        x = torch.relu(self.bn1(self.fc1(combined)))
        x = self.dropout(x)
        output = torch.sigmoid(self.fc2(x))
        return output.squeeze(1)
