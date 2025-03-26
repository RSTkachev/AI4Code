import torch
import torch.nn as nn

from transformers import AutoModel


class OrderPredictionModel(nn.Module):
    def __init__(self, model_name, hidden_dim, dropout_prob=0.2):
        super(OrderPredictionModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.type_embedding = nn.Embedding(2, 32)  

        self.proj = nn.Linear(768, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32 * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, input_ids1, att_mask1, cell_type1, input_ids2, att_mask2, cell_type2):
        with torch.no_grad():
            output1 = self.model(input_ids1, attention_mask=att_mask1).last_hidden_state
            output2 = self.model(input_ids2, attention_mask=att_mask2).last_hidden_state
        
        embedding1 = torch.mean(output1, dim=1)
        embedding2 = torch.mean(output2, dim=1)

        embedding1 = self.proj(embedding1)
        embedding2 = self.proj(embedding2)

        type_emb1 = self.type_embedding(cell_type1)
        type_emb2 = self.type_embedding(cell_type2)

        combined = torch.cat([embedding1, type_emb1, embedding2, type_emb2], dim=1)

        output = torch.sigmoid(self.classifier(combined))
        return output.squeeze(1)
