import os
import datetime
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score
import copy
from data import SequenceClassificationDataset
from models import BertHSLN
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from train import testing_step_cross
from utils import save_model_state_dict, load_config, load_checkpoint
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

BERT_MODEL = "bert-base-uncased"

parser = argparse.ArgumentParser(description="Run DL experiments.")
parser.add_argument("config", type=str, help="Name of the config.json file to be loaded.")
args = parser.parse_args()
config_file = args.config


# Load the configuration from the JSON file
mconfig = load_config(config_file)
print(f'loaded config file {mconfig}')
paheli = False
if mconfig['dataset']=='paheli':
    paheli=True


root = os.path.dirname(os.path.realpath(__file__))
paheli_path = root + f"/datasets/paheli/"
mit_path = root + f"/datasets/malik_it/"
mcl_path = root + f"/datasets/malik_cl/"
p_ds = SequenceClassificationDataset(Path(paheli_path, 'test.json'), 'paheli')
mit_ds = SequenceClassificationDataset(Path(mit_path, 'test.json'), 'malik_it')
mcl_ds = SequenceClassificationDataset(Path(mcl_path, 'test.json'), 'malik_cl')

dataloaders = {}
dataloaders['pahelis'] = DataLoader(p_ds, batch_size=1, shuffle=True)
dataloaders['malik_it'] = DataLoader(mit_ds, batch_size=1, shuffle=True)
dataloaders['malik_cl'] = DataLoader(mcl_ds, batch_size=1, shuffle=True)

save_path = f"/srv/sarwath/{mconfig['model_name']}.pth"
print("Model Name:",mconfig['model_name'])
print('*'*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_checkpoint(save_path,mconfig, device)
preds = testing_step_cross(model, dataloaders, device,paheli)
for k,v in preds.items():
    print('_-_'*40)
    print(f"macro_f1 for {k} = {f1_score(v['true_labels'], v['predictions'], average='macro')*100} ")
# print(f"""Macro f1: {f1_score(true_labels, predictions, average='macro')*100} 
#       Micro f1: {f1_score(true_labels, predictions, average='micro')*100}
#       Weighted f1: {f1_score(true_labels, predictions, average='weighted')*100}""")
