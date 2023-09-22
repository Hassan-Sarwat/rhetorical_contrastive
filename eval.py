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
from train import training_step, validation_step, testing_step
from utils import save_model_state_dict, load_config, load_checkpoint
import argparse
import json

BERT_MODEL = "bert-base-uncased"


parser = argparse.ArgumentParser(description="Run DL experiments.")
parser.add_argument("config", type=str, help="Name of the config.json file to be loaded.")
args = parser.parse_args()
config_file = args.config

# Load the configuration from the JSON file
mconfig = load_config(config_file)
print(f'loaded config file {mconfig}')

root = os.path.dirname(os.path.realpath(__file__))
dataset_path = root + f"/datasets/{mconfig['dataset']}/"
print(dataset_path)
print('*'*40)
if mconfig['dataset']=='build':
      test_dataset = SequenceClassificationDataset(Path(dataset_path, 'dev.json'), mconfig['dataset'])
else:
      test_dataset = SequenceClassificationDataset(Path(dataset_path, 'test.json'), mconfig['dataset'])
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)


save_path = f"/srv/sarwath/{mconfig['model_name']}.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_checkpoint(save_path,mconfig, device)
test_loss, test_accuracy, predictions, true_labels = testing_step(model, test_dataloader, device)
print(f"""Macro f1: {f1_score(true_labels, predictions, average='macro')*100} 
      Micro f1: {f1_score(true_labels, predictions, average='micro')*100}
      Weighted f1: {f1_score(true_labels, predictions, average='weighted')*100}""")
