import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)



class TextDataset(Dataset):
    def __init__(self, tokenizer, args, df):
    

        self.examples = df[["cell_id","source"]].values
        self.args = args
        self.tokenizer = tokenizer
        logger.info("Total number: {}".format(len(self.examples)))   

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        args = self.args
        tokenizer = self.tokenizer        
        idx,code = self.examples[i]
        
        code_tokens = tokenizer.tokenize(code)[:args.length-4]
        code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.length - len(code_ids)
        code_ids += [tokenizer.pad_token_id]*padding_length
            
        return torch.tensor(code_ids)

                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default="tmp/feature-epoch10", type=str)
    parser.add_argument("--data_dir", default="tmp/data", type=str)
    parser.add_argument("--model_path", default="../pre-train/saved_models/len128-num32-epoch20/epoch-10-0.0399", type=str)
    parser.add_argument("--length", default=128, type=int)    
    parser.add_argument("--do_train", action='store_true')  
    parser.add_argument("--eval_batch_size", default=1024, type=int)
    
    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) 
        
    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    config = RobertaConfig.from_pretrained(args.model_path)
    model = RobertaModel.from_pretrained(args.model_path) 
    logger.info("Training/evaluation parameters %s", args)
    
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
    model.eval()
    
    dfs = []
    if args.do_train:
        train_df = pd.read_pickle(args.data_dir + "/train.pkl")
        dfs.append((train_df,"train"))
        val_df = pd.read_pickle(args.data_dir + "/val.pkl")
        dfs.append((val_df,"val"))
        
    test_df = pd.read_pickle(args.data_dir + "/test.pkl")
    dfs.append((test_df,"test"))        
    
    for df,prefix in dfs:
        dataset = TextDataset(tokenizer, args, df)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,num_workers=4)
        
        
        
        code_vecs = [] 
        for batch in tqdm(dataloader):  
            source_ids = batch.to(args.device)
            with torch.no_grad():
                mask = source_ids.ne(config.pad_token_id)
                token_embeddings = model(source_ids,attention_mask = mask.unsqueeze(1) * mask.unsqueeze(2))[0]
                sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
            code_vecs.append(sentence_embeddings.cpu().numpy().astype(np.float16))
        code_vecs = np.concatenate(code_vecs,0)
        dic = {}
        for x,y in zip(dataset.examples,code_vecs):
            dic[x[0]] = y
        pickle.dump(dic,open(args.output_dir+"/{}.pkl".format(prefix),"wb"))
