import torch
from torch import nn
from transformers import AutoModel, BertModel


class ListWiseOrderPredictionModel(nn.Module):

    def __init__(self, hidden_dim=768, dropout_prob=0.1):
        super().__init__()
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.bert_text = BertModel.from_pretrained("bert-base-multilingual-uncased")
        self.type_embedding = nn.Embedding(2, 8)

        self.proj = nn.Linear(768 + 8, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, cell_types):
        device = input_ids.device
        N = input_ids.size(0)

        code_mask = (cell_types == 1)
        text_mask = (cell_types == 0)

        embeddings = torch.zeros(N, 768, device=device, dtype=torch.float32)

        if code_mask.any():
            code_idx = code_mask.nonzero(as_tuple=True)[0]
            out_code = self.codebert(
                input_ids[code_idx],
                attention_mask=attention_mask[code_idx]
            ).pooler_output
            embeddings[code_idx] = out_code

        if text_mask.any():
            text_idx = text_mask.nonzero(as_tuple=True)[0]
            out_text = self.bert_text(
                input_ids[text_idx],
                attention_mask=attention_mask[text_idx]
            ).pooler_output
            embeddings[text_idx] = out_text

        type_emb = self.type_embedding(cell_types)

        x = torch.cat([embeddings, type_emb], dim=1)
        x = self.dropout(self.act(self.proj(x)))
        scores = self.classifier(x)
        return scores.squeeze(-1)
