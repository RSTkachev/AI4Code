import os
import gc
import torch
import logging
import argparse
import pickle
import random
import json

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer
from torch.utils.data import Dataset

pd.options.display.width = 180
pd.options.display.max_colwidth = 120

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
# File information
parser.add_argument('--output_dir', type=str, default="tmp/data")
parser.add_argument('--data_dir', type=str, default="/home/t-dayaguo/AI4Code")


# Mode information
parser.add_argument("--do_train", action='store_true')
args = parser.parse_args()
logger.info(args)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir) 

def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )
if args.do_train:
    ###############################################################################
    paths_train = list((Path(args.data_dir) / 'train').glob('*.json'))
    notebooks_train = [
        read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
    ]
    df = (
        pd.concat(notebooks_train)
        .set_index('id', append=True)
        .swaplevel()
        .sort_index(level='id', sort_remaining=False)
    )
    ###############################################################################
    def get_ranks(base, derived):
        return [base.index(d) for d in derived]

    df_orders = pd.read_csv(
        Path(args.data_dir) / 'train_orders.csv',
        index_col='id',
        squeeze=True,
    ).str.split()

    df_orders_ = df_orders.to_frame().join(
        df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
        how='right',
    )

    ranks = {}
    for id_, cell_order, cell_id in df_orders_.itertuples():
        ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

    df_ranks = (
        pd.DataFrame
        .from_dict(ranks, orient='index')
        .rename_axis('id')
        .apply(pd.Series.explode)
        .set_index('cell_id', append=True)
    )
    ###############################################################################
    df_ancestors = pd.read_csv(Path(args.data_dir) / 'train_ancestors.csv', index_col='id')
    df = df.reset_index().merge(df_ranks, on=["id", "cell_id"]).merge(df_ancestors, on=["id"])
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=2022)

    train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))

    train_df = df.loc[train_ind].reset_index(drop=True)
    val_df = df.loc[val_ind].reset_index(drop=True)
    ###############################################################################
    train_df = train_df[["id","cell_id","cell_type","source","rank"]].set_index('id', append=True).swaplevel().sort_index(level='id', sort_remaining=False)
    val_df = val_df[["id","cell_id","cell_type","source","rank"]].set_index('id', append=True).swaplevel().sort_index(level='id', sort_remaining=False)
    train_df.to_pickle(args.output_dir + "/train.pkl")
    val_df.to_pickle(args.output_dir + "/val.pkl")

        
        
paths_test = list((Path(args.data_dir) / 'test').glob('*.json'))
notebooks_test = [
    read_notebook(path) for path in tqdm(paths_test, desc='Test NBs')
]
test_df = (
    pd.concat(notebooks_test)
    .set_index('id', append=True)
    .swaplevel()
    .sort_index(level='id', sort_remaining=False)
).reset_index()
test_df = test_df.set_index('id', append=True).swaplevel().sort_index(level='id', sort_remaining=False)
test_df.to_pickle(args.output_dir + "/test.pkl")