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
from utils import save_model_state_dict, load_config
import argparse
import json

seed_val = 42
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


BERT_MODEL = "bert-base-uncased"


parser = argparse.ArgumentParser(description="Run DL experiments.")
parser.add_argument("config", type=str, help="Name of the config.json file to be loaded.")
args = parser.parse_args()
config_file = args.config

# Load the configuration from the JSON file
mconfig = load_config(config_file)
print(f'loaded config file {mconfig}')


# LOAD DATA
root = os.path.dirname(os.path.realpath(__file__))
dataset_path = root + f"/datasets/{mconfig['dataset']}/"
print('dataset_path:',dataset_path)
train_dataset = SequenceClassificationDataset(Path(dataset_path,'train.json'), mconfig['dataset'])
dev_dataset = SequenceClassificationDataset(Path(dataset_path, 'dev.json'), mconfig['dataset'])
if mconfig['dataset']=='build':
    test_dataset = dev_dataset
else:
    test_dataset = SequenceClassificationDataset(Path(dataset_path, 'test.json'), mconfig['dataset'])

print(f" size of train:{len(train_dataset)}, size of dev:{len(dev_dataset)}, size of test:{len(test_dataset)}")

# INIT DATA LOADERS
batch_size=1
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# INIT MODEL
print('Cude is available:',torch.cuda.is_available())
model = BertHSLN(mconfig, num_labels = mconfig['nlabels'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device used: {device}")
model.to(device)

# INIT OPTIMIZER
optimizer = Adam(model.parameters(), lr=mconfig['lr'])
epoch_scheduler = StepLR(optimizer, step_size=1, gamma=mconfig["lr_epoch_decay"])

# START TRAINING
train_epoch_losses = {'loss': [], 'cls':[], 'pc':[], 'sc':[],'cont':[]}
train_epoch_acc = []
train_epoch_f1 = []

dev_epoch_losses = []
dev_epoch_acc = []
dev_epoch_f1 = []


best_dev_f1 = 0
best_model = None

epochs = mconfig['max_epochs']

# Training loop
for epoch in range(epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))

    # # Training
    train_loss = training_step(model, optimizer, epoch_scheduler, train_dataloader, device)
    train_epoch_losses['loss'].append(train_loss['loss'])
    train_epoch_losses['cls'].append(train_loss['cls'])
    train_epoch_losses['pc'].append(train_loss['pc'])
    train_epoch_losses['sc'].append(train_loss['sc'])
    train_epoch_losses['cont'].append(train_loss['cont'])
    print('*'*60)
    print(f"loss {train_loss['loss']} cls_loss {train_loss['cls']}, pc {train_loss['pc']}, sc {train_loss['sc']}")
    if mconfig['supervised_loss']:
        print(f"Epoch {epoch+1}/{epochs} - Training contrastive Loss: {train_loss['cont']:.4f}")

    # Validation
    dev_f1, dev_loss = validation_step(model, valid_dataloader, device)
    dev_epoch_losses.append(dev_loss)
    dev_epoch_f1.append(dev_f1)
    print(f"Epoch {epoch+1}/{epochs} - dev loss {dev_loss} - F1 {dev_f1}")
    test_loss, test_accuracy, predictions, true_labels = testing_step(model, test_dataloader, device)
    test_f1 = f1_score(predictions, true_labels, average='macro')

    if dev_f1 > best_dev_f1:
        best_dev_f1 = dev_f1
        best_model = copy.deepcopy(model)
        print(f'saving the checkpoint with dev f1 - {dev_f1}')

# TESTING
test_loss, test_accuracy, predictions, true_labels = testing_step(best_model, test_dataloader, device)
test_f1 = f1_score(predictions, true_labels, average='macro')

# SAVE BEST MODEL
today = datetime.datetime.now().strftime('%Y/%m/%d')
save_path = f"/srv/sarwath/{mconfig['model_name']}_{np.round(test_f1, 4)}.pth"
config_path = f"/srv/sarwath/{mconfig['model_name']}_{np.round(test_f1, 4)}.json"
print('save_path:',save_path)
save_model_state_dict(best_model, save_path)
with open(config_path, "w") as outfile:
    json.dump(mconfig, outfile)