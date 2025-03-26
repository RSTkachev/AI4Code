import torch.nn as nn
import torch    
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

class Model(nn.Module):   
    def __init__(self,args):
        super(Model, self).__init__()
        self.args = args
        self.config = RobertaConfig()
        self.config.hidden_size += self.config.num_attention_heads
        self.config.num_hidden_layers = 6
        self.config.vocab_size = 5
        self.config.type_vocab_size = 3
        self.config.max_position_embeddings = 2048

        self.transformer = RobertaModel(self.config)
        self.proj1 = nn.Linear(self.config.hidden_size, self.config.hidden_size*4)
        self.proj2 = nn.Linear(self.config.hidden_size*4,self.config.hidden_size )
        self.linear = nn.Linear(self.config.hidden_size, 1)
        self.classifier_linear = nn.Linear(self.config.hidden_size, self.args.code_number+1)
        self.code_insert_linear = nn.Linear(self.config.hidden_size, 1)
        
    def forward(self, features, cell_types, cell_ids, positions, pivots, ranks=None ,classifier_target=None,code_insert_target=None,weight=None): 
        
        features = torch.cat((features,pivots[:,:,None].repeat(1,1,self.config.num_attention_heads)),-1)
        features = self.transformer(inputs_embeds=features, 
                                    attention_mask=cell_ids.ne(-1), 
                                    token_type_ids=cell_types, 
                                    position_ids=positions)[0]
        features = torch.tanh(self.proj1(features))
        features = torch.tanh(self.proj2(features))
        scores = torch.sigmoid(self.linear(features))[:,:,0]
        classifier_scores = self.classifier_linear(features)
        code_insert_scores = self.code_insert_linear(features)
        
        if ranks is not None:
            mask = ranks.reshape(-1).ne(-1) & cell_types.reshape(-1).eq(1)
            #print(scores.shape)
            #print(weight.shape)
            #print(ranks.shape)
            loss = (((scores.reshape(-1)*weight.reshape(-1))[mask] - (ranks.reshape(-1)*weight.reshape(-1))[mask]).abs()).sum()/weight.reshape(-1)[mask].sum()
            batch_dim=features.shape[0]
            batch_mask=(ranks.ne(-1) & cell_types.eq(1)).unsqueeze(-1).repeat((1,1,self.args.code_number+1))
            classifier_loss = F.cross_entropy(classifier_scores[batch_mask].reshape(-1,self.args.code_number+1),classifier_target.reshape(-1)[mask])
            
            code_insert_mask = cell_types.reshape(-1).eq(0)
            code_insert_loss = ((code_insert_scores.reshape(-1)[code_insert_mask] - code_insert_target.reshape(-1)[code_insert_mask]).abs()).mean()
            
            return loss,classifier_loss+4*code_insert_loss,scores 
        else:
            return scores 
