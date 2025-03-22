import torch
import torch.nn as nn

from transformers import BertModel, AutoModel


class OrderPredictionModel(nn.Module):
    def __init__(self, hidden_dim, dropout_prob=0.1):
        super(OrderPredictionModel, self).__init__()

        self.bert_text = BertModel.from_pretrained("bert-base-multilingual-uncased")
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")

        self.type_embedding = nn.Embedding(2, 8)
        self.fc1 = nn.Linear(768 * 2 + 8 * 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, 1)

    # def forward(self, input_ids1, att_mask1, cell_type1, input_ids2, att_mask2, cell_type2):
    #     with torch.no_grad():
    #         embedding1 = self.bert(input_ids1, attention_mask=att_mask1).pooler_output
    #         embedding2 = self.bert(input_ids2, attention_mask=att_mask2).pooler_output

    #     type_emb1 = self.type_embedding(cell_type1)
    #     type_emb2 = self.type_embedding(cell_type2)

    #     combined = torch.cat([embedding1, type_emb1, embedding2, type_emb2], dim=1)
    #     x = torch.relu(self.bn1(self.fc1(combined)))
    #     x = self.dropout(x)
    #     output = torch.sigmoid(self.fc2(x))
    #     return output.squeeze(1)

    @staticmethod
    def _get_batch_embeddings(input_ids, attention_mask, cell_type, code_model, text_model):

        device = input_ids.device
        batch_size = input_ids.size(0)

        hidden_size = code_model.config.hidden_size

        embeddings = torch.zeros(batch_size, hidden_size, device=device, dtype=torch.float32)

        code_mask = (cell_type == 1)
        text_mask = (cell_type == 0)

        if code_mask.any():
            code_indices = code_mask.nonzero(as_tuple=True)[0]
            code_input_ids = input_ids[code_indices]
            code_attention_mask = attention_mask[code_indices]

            out_code = code_model(code_input_ids, attention_mask=code_attention_mask).pooler_output
            embeddings[code_indices] = out_code

        if text_mask.any():
            text_indices = text_mask.nonzero(as_tuple=True)[0]
            text_input_ids = input_ids[text_indices]
            text_attention_mask = attention_mask[text_indices]

            out_text = text_model(text_input_ids, attention_mask=text_attention_mask).pooler_output
            embeddings[text_indices] = out_text

        return embeddings

    def forward(self, input_ids1, att_mask1, cell_type1, input_ids2, att_mask2, cell_type2):

        embedding1 = self._get_batch_embeddings(input_ids1, att_mask1, cell_type1, 
                                                code_model=self.codebert,text_model=self.bert_text)

        embedding2 = self._get_batch_embeddings(input_ids2, att_mask2, cell_type2, code_model=self.codebert, text_model=self.bert_text)

        type_emb1 = self.type_embedding(cell_type1)
        type_emb2 = self.type_embedding(cell_type2)

        combined = torch.cat([embedding1, type_emb1, embedding2, type_emb2], dim=1)
        x = torch.relu(self.bn1(self.fc1(combined)))
        x = self.dropout(x)
        output = torch.sigmoid(self.fc2(x))

        return output.squeeze(1)