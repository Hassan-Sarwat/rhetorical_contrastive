from allennlp.common.util import pad_sequence_to_length
import os
import random
import time
import datetime
import json
import pandas as pd
import numpy as np
import torch
import pickle
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
import functools
import operator
from sklearn.metrics import classification_report

def batch_to_tensor(b):
    # convert to dictionary of tensors and pad the tensors
    max_sentence_len = 128
    result = {}
    for k, v in b.items():

        if k in ["input_ids", "attention_mask"]:
            # determine the max sentence len in the batch
            max_sentence_len = -1
            for sentence in v:
                sentence = torch.cat(sentence)
                max_sentence_len = max(len(sentence), max_sentence_len)
            # pad the sentences to max sentence len
            for i, sentence in enumerate(v):
                v[i] = pad_sequence_to_length(sentence, desired_length=max_sentence_len)
        if k!='doc_name' and k!= 'label_ids':
            result[k] = torch.tensor(v).unsqueeze(0)
        elif k == 'label_ids':
            result[k] = torch.tensor(v)
        else:
            result[k] = v
    return result


def training_step(model, optimizer, scheduler, data_loader, device, crf = False):
    model.train()  # Set the model to train mode
    train_loss = {'sc':0, 'pc':0, 'cls': 0, 'loss':0,'cont':0,'d':0}
    train_correct = 0
    train_total = 0

    all_labels = []
    all_predicted = []

    for batch_idx, batch in enumerate(data_loader):
        batch = batch_to_tensor(batch)
        # handle an empty batch --> error in data preparation
        if batch["input_ids"].shape[1] == 0:
          print("Skipping an empty batch.")
          continue

        optimizer.zero_grad()
        for key, tensor in batch.items():
            batch[key] = tensor.to(device)
        #print(batch['input_ids'].shape)
        labels = batch['label_ids']
        #print(labels.shape)
        #optimizer.zero_grad()

        # Forward pass
        outputs, embeddings = model(batch, labels, get_embeddings=True)

        

        # Calculate loss
        loss = outputs['loss']
        train_loss['loss'] = train_loss['loss'] + loss.detach().item()
        cont_loss,sc_loss, pc_loss, cls_loss,d_loss = 0,0,0,0,0
        if model.sl:
            cont_loss = outputs['cont_loss'] 
            train_loss['cont'] = train_loss['cont'] + cont_loss.detach().item()
        if model.sp:
            sc_loss = outputs['sc_loss']
            pc_loss = outputs['pc_loss']
            cls_loss = outputs['cls_loss']
            train_loss['sc'] = train_loss['sc'] + sc_loss.detach().item()
            train_loss['pc'] = train_loss['pc'] + pc_loss.detach().detach().item()
            train_loss['cls'] = train_loss['cls'] + cls_loss.detach().detach().item()
        if model.mp:
            sc_loss = outputs['sc_loss'] 
            cls_loss = outputs['cls_loss']
            d_loss = outputs['d_loss']
            train_loss['sc'] = train_loss['sc'] + sc_loss.detach().item()
            train_loss['cls'] = train_loss['cls'] + cls_loss.detach().detach().item()
            train_loss['d'] = train_loss['d'] + cls_loss.detach().detach().item()

        if batch_idx % 10 == 0:
          print(f'after {batch_idx} step: loss {loss}, cont_loss {cont_loss}\n, cls_loss {cls_loss}, pc_loss {pc_loss}, sc_loss {sc_loss}, d_loss {d_loss}')
          



        # Backward pass and optimization
        loss =  loss + pc_loss + sc_loss + cls_loss + cont_loss + d_loss
        loss.backward()
        # cont_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        

    scheduler.step()
    # Calculate epoch statistics
    train_loss['loss'] = train_loss['loss'] / len(data_loader)
    if model.sl:
        train_loss['cont'] = train_loss['cont'] / len(data_loader)
    if model.sp:
        train_loss['cls'] = train_loss['cls'] / len(data_loader)
        train_loss['pc'] = train_loss['pc'] / len(data_loader)
        train_loss['sc'] = train_loss['sc'] / len(data_loader)
    if model.mp:
        train_loss['cls'] = train_loss['cls'] / len(data_loader)
        train_loss['sc'] = train_loss['sc'] / len(data_loader)
        train_loss['d'] = train_loss['d']/len(data_loader)
    return train_loss


def validation_step(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    dev_loss = 0.0
    dev_correct = 0
    dev_total = 0
    all_labels = []
    all_predicted = []


    for batch in data_loader:
        # handle an empty batch --> error in data preparation
        with torch.no_grad():
            batch = batch_to_tensor(batch)
            if batch["input_ids"].shape[1] == 0:
              continue
            for key, tensor in batch.items():
               batch[key] = tensor.to(device)
            labels = batch['label_ids']
            # Forward pass
            outputs = model(batch)


            # Calculate loss
            
            logits = outputs['logits'].squeeze()
            dev_loss += F.cross_entropy(logits, labels.squeeze()).item()
            predicted_label = outputs['predicted_label']
            
            
            # save all epoch labels and predicted labels
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted_label.cpu().numpy())

    # Calculate epoch statistics
    print(f"len predicted:{len(all_predicted)} len labels: {len(all_labels)}")

    #all_predicted = functools.reduce(operator.iconcat, all_predicted, [])
    all_labels = functools.reduce(operator.iconcat, all_labels, [])
   
    f1 = f1_score(all_labels, all_predicted, average='macro')
    dev_loss /= len(data_loader)
    return f1, dev_loss



    from sklearn.metrics import classification_report



def testing_step(model, data_loader, device,):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    model.to(device)

    for batch in data_loader:
        with torch.no_grad():

            batch = batch_to_tensor(batch)
            for key, tensor in batch.items():
               batch[key] = tensor.to(device)
            labels = batch['label_ids']
            # Forward pass
            outputs = model(batch)
            #max_sequence_length = outputs.size(1)
            #tlengths = torch.tensor(lengths).unsqueeze(1)
            #mask = torch.arange(max_sequence_length).unsqueeze(0).to(device) < tlengths
            #masked_output = outputs[mask]
            logits = outputs['logits'].squeeze()
            predicted_labels = outputs['predicted_label']

            # Calculate loss
            #loss = criterion(masked_output, labels)
            #test_loss += loss.item()

            # Store predictions and true labels
            #predicted_labels = masked_output.argmax(dim=-1)
            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            # Calculate accuracy
            predicted_labels = predicted_labels.to('cpu')
            labels = labels.to('cpu')

            correct = (predicted_labels == labels).sum().item()
            test_correct += correct
            test_total += labels.shape[0]

    #predictions = functools.reduce(operator.iconcat, predictions, [])
    true_labels = functools.reduce(operator.iconcat, true_labels, [])

    test_loss /= len(data_loader)
    test_accuracy = test_correct / test_total

    print(f"Testing Loss: {test_loss:.4f} - Testing Accuracy: {test_accuracy:.4f}")
    print(classification_report(true_labels, predictions))
    return test_loss, test_accuracy, predictions, true_labels


def mapper(data, model_name, data_name):
    p_m = {0:1,1:6,2:0,3:3,4:4,5:5,6:2}
    m_p = {0:2,1:0,2:6,3:3,4:4,5:5,6:1}
    if model_name == 'paheli' and data_name == 'malik':
        new_data = [p_m[i.item()] for i in data]
    else:
        new_data = [m_p[i.item()] for i in data]
    return new_data


def testing_step_cross(model, dataloaders, device, paheli):
    model.eval()  # Set the model to evaluation mode
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    preds = {}
    model_name = 'malik'
    if paheli:
        model_name = 'paheli'
    model.to(device)
    for k in dataloaders.keys():
        data_loader = dataloaders[k]
        preds[k] = {}
        predictions = []
        true_labels = []
        print(f"Predicting for {k} data using {model_name} model")
        print('x'*100)
        for batch in data_loader:
            with torch.no_grad(): 
                batch = batch_to_tensor(batch)
                for key, tensor in batch.items():
                    batch[key] = tensor.to(device)
                labels = batch['label_ids']
                # Forward pass
                outputs = model(batch)
                #max_sequence_length = outputs.size(1)
                #tlengths = torch.tensor(lengths).unsqueeze(1)
                #mask = torch.arange(max_sequence_length).unsqueeze(0).to(device) < tlengths
                #masked_output = outputs[mask]
                changed = False
                logits = outputs['logits'].squeeze()
                predicted_labels = outputs['predicted_label']
                if  paheli and 'malik' in k:
                    predicted_labels = mapper(predicted_labels,'paheli','malik')
                    print('Changing paheli predictions to malik')
                    print('-'*100)
                    changed = True
                elif paheli == False and 'paheli' in k:
                    predicted_labels = mapper(predicted_labels,'malik','paheli')
                    print('Changing Malik labels to paheli')
                    print('-'*100)
                    changed = True

                # Calculate loss
                #loss = criterion(masked_output, labels)
                #test_loss += loss.item()

                # Store predictions and true labels
                #predicted_labels = masked_output.argmax(dim=-1)
                if changed:
                    predictions.extend(predicted_labels)
                else:
                    predictions.extend(predicted_labels.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        #predictions = functools.reduce(operator.iconcat, predictions, [])
        true_labels = functools.reduce(operator.iconcat, true_labels, [])
        preds[k]['true_labels'] = true_labels
        preds[k]['predictions'] = predictions
        
    return preds

