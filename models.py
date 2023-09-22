from allennlp.common.util import pad_sequence_to_length
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn.util import masked_mean, masked_softmax
import copy

from transformers import BertModel

from allennlp.modules import ConditionalRandomField
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np


class MultiProtoSimModel(nn.Module):

    def __init__(self, rhetorical_role, embedding_width, num_prototypes_per_class):
        super(ProtoSimModel, self).__init__()
        
        # Initialize multiple prototypes per class
        total_prototypes = rhetorical_role * num_prototypes_per_class
        self.prototypes = nn.Embedding(total_prototypes, embedding_width)
        
        self.classification_layer = nn.Linear(embedding_width, rhetorical_role)
        self.cross_entropy = torch.nn.CrossEntropyLoss()



    def forward(self, role_embedding, role_id):
        # Select the set of prototypes for the given role_id
        start_idx = role_id * self.num_prototypes_per_class
        end_idx = (start_idx + self.num_prototypes_per_class)
        protos = self.prototypes(torch.arange(start_idx, end_idx).to(role_id.device))

        # Normalize the prototypes and role embeddings
        protos = F.normalize(protos, p=2, dim=-1)
        role_embedding = F.normalize(role_embedding, p=2, dim=-1)

        # Calculate distances to all prototypes
        similarity = torch.sum(protos * role_embedding.unsqueeze(1), dim=-1)
        similarity = torch.exp(similarity)
        dist = 1 - 1 / (1 + similarity)

        # Find the minimum distance and the index of the nearest prototype
        min_dist, min_idx = torch.min(dist, dim=1)

        # Classification prediction using only the nearest prototype
        nearest_proto = protos[min_idx]
        predict_role = self.classification_layer(nearest_proto)

        return min_dist, predict_role

    
    def get_diversity_loss(self):
        diversity_loss = 0.0
        num_classes = self.prototypes.weight.shape[0] // self.num_prototypes_per_class  # Assuming self.num_prototypes_per_class is defined

        for i in range(num_classes):
            # Select the prototypes for the current class
            start_idx = i * self.num_prototypes_per_class
            end_idx = (i + 1) * self.num_prototypes_per_class
            class_prototypes = self.prototypes.weight[start_idx:end_idx]

            # Normalize the prototypes (if you're using cosine similarity)
            class_prototypes = F.normalize(class_prototypes, p=2, dim=-1)

            # Calculate pairwise distances
            # Using cosine similarity here; you can use other distance metrics
            similarity_matrix = torch.matmul(class_prototypes, class_prototypes.t())

            # Remove diagonal elements as we don't want to penalize similarity with oneself
            similarity_matrix = similarity_matrix - torch.eye(self.num_prototypes_per_class).to(similarity_matrix.device)

            # Compute the loss for this class; we want to minimize similarity hence -1.0 * mean(similarity)
            diversity_loss -= torch.mean(similarity_matrix)

        # Average the loss over all classes
        diversity_loss /= num_classes

        return diversity_loss
    

    def get_classification_loss(self, embeddings, labels):
        batch_size = embeddings.size(0)
        cls_loss = 0.0
        unique_labels = torch.unique(labels)

        for label in unique_labels:
            label_mask = labels == label
            other_mask = labels != label

            label_embeddings = embeddings[label_mask]
            other_embeddings = embeddings[other_mask]
            other_labels = labels[other_mask]

            # Get the minimum distance and the classification score of the nearest prototype
            _, p_predicted_role = self.forward(label_embeddings, label)
            _, n_predicted_role = self.forward(other_embeddings, other_labels)

            p_label = label.repeat(p_predicted_role.size(0)).type(torch.LongTensor).to(p_predicted_role.device)

            cls_loss += self.cross_entropy(p_predicted_role, p_label)
            cls_loss += self.cross_entropy(n_predicted_role, other_labels)

        cls_loss /= batch_size
        return cls_loss
    
    def get_sample_centric_loss(self, embeddings, labels):
        batch_size = embeddings.size(0)
        cluster_loss = 0.0
        unique_labels = torch.unique(labels)

        for label in unique_labels:
            label_mask = labels == label
            label_embeddings = embeddings[label_mask]

            # Calculate psim: distance between embeddings and their nearest prototype within the class
            p_sim, _ = self.forward(label_embeddings, label)

            # Calculate nsim: distance between embeddings and nearest prototypes of different classes
            other_labels = unique_labels[unique_labels != label]
            n_sim_list = []
            for other_label in other_labels:
                n_sim, _ = self.forward(label_embeddings, other_label)
                n_sim_list.append(n_sim)

            n_sim = torch.min(torch.stack(n_sim_list), dim=0)[0]

            cluster_loss += -(torch.mean(torch.log(p_sim + 1e-5)) + torch.mean(torch.log(1 - n_sim + 1e-5)))

        cluster_loss /= batch_size

        return cluster_loss


class ProtoSimModel(nn.Module):

    def __init__(self, rhetorical_role, embedding_width):
        nn.Module.__init__(self)
        self.prototypes = nn.Embedding(rhetorical_role, embedding_width)
        self.classification_layer = nn.Linear(embedding_width, rhetorical_role)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, role_embedding, role_id):
        protos = self.prototypes(role_id)
        
        protos = F.normalize(protos, p=2, dim=-1)  # Normalize prototype embeddings
        role_embedding = F.normalize(role_embedding, p=2, dim=-1)  # Normalize input embeddings
        
        similarity = torch.sum(protos * role_embedding, dim=-1)  # Cosine similarity
        similarity = torch.exp(similarity)
        dist = 1 - 1 / (1 + similarity)  # Cosine distance
        
        predict_role = self.classification_layer(protos)
        
        return dist, predict_role

    def get_proto_centric_loss(self, embeddings, labels):
        """
        prototypes centric view
        """
        batch_size = embeddings.size(1)
        cluster_loss = 0.0

        for label in torch.unique(labels):
            label_mask = labels == label
            other_mask = labels != label


            label_embeddings = embeddings[label_mask]
            other_embeddings = embeddings[other_mask]
            other_labels = labels[other_mask]  # Capture the labels for other embeddings

            p_sim, _ = self.forward(label_embeddings, label)
            n_sim, _ = self.forward(other_embeddings, label)

            cluster_loss += -(torch.mean(torch.log(p_sim + 1e-5)) + torch.mean(torch.log(1 - n_sim + 1e-5)))

        cluster_loss /= batch_size

        return cluster_loss


    def get_classification_loss(self, embeddings, labels):
        batch_size = embeddings.size(1)
        cls_loss = 0.0

        for label in torch.unique(labels):
            label_mask = labels == label
            other_mask = labels != label

            label_embeddings = embeddings[label_mask]
            other_embeddings = embeddings[other_mask]
            other_labels = labels[other_mask]

            _, p_predicted_role = self.forward(label_embeddings, label)
            _, n_predicted_role = self.forward(other_embeddings, other_labels)

            p_label = label.repeat(p_predicted_role.size(0)).type(torch.FloatTensor).cuda()

            cls_loss += self.cross_entropy(p_predicted_role, p_label)
            cls_loss += self.cross_entropy(n_predicted_role, other_labels)

        cls_loss /= batch_size
        return cls_loss
    
    def get_sample_centric_loss(self, embeddings, labels):
        """
        sample centric view
        """
        batch_size = embeddings.size(1)
        cluster_loss = 0.0

        unique_labels = torch.unique(labels)

        for label in unique_labels:
            label_mask = labels == label
            label_embeddings = embeddings[label_mask]

            # Calculate psim: distance between embeddings and their corresponding prototype
            p_sim, _ = self.forward(label_embeddings, label)

            # Calculate nsim: distance between embeddings and prototypes of different classes
            other_labels = unique_labels[unique_labels != label]
            n_sim_list = []
            for other_label in other_labels:
                n_sim, _ = self.forward(label_embeddings, other_label)
                n_sim_list.append(n_sim)

            n_sim = torch.mean(torch.stack(n_sim_list), dim=0)

            cluster_loss += -(torch.mean(torch.log(p_sim + 1e-5)) + torch.mean(torch.log(1 - n_sim + 1e-5)))

        cluster_loss /= batch_size

        return cluster_loss

class SupConLoss(nn.Module):

    def __init__(self, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.contrast_mode = contrast_mode

    def forward(self, features, labels,memory_features=None, memory_labels=None, weighted = False):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if memory_features==None and memory_labels == None:
          #print(labels.size())
          labels = labels.contiguous().view(-1, 1)
          #print(labels.size())
          anchor_feature = features
          mask = torch.eq(labels, labels.T).float().to(device)
          anchor_count = features.shape[0]
          contrast_count = anchor_count
          contrast_feature = anchor_feature
          logits_mask = torch.ones_like(mask).to(device)
          self_contrast_mask = 1 - torch.diag(torch.ones((mask.size()[0])))
          logits_mask[:,:mask.size()[0]] = logits_mask[:,:mask.size()[0]].clone() * self_contrast_mask.to(device)
        elif memory_features!=None and  memory_labels!=None:
          anchor_count = features.shape[0]
          anchor_feature = features
          labels = labels.contiguous().view(-1, 1)
          memory_labels = memory_labels.contiguous().view(-1, 1)
          memory_count = memory_features.size()[0]
          contrast_count = anchor_count + memory_features.size()[0]
          contrast_labels = torch.cat([labels,memory_labels])
          mask = torch.eq(labels, contrast_labels.T).float().to(device)
          positive_mask = torch.eq(labels, labels.T).float().to(device)
          #filter_mask = torch.zeros((anchor_count, memory_count))
          #mask = torch.cat((positive_mask, filter_mask.to(device)), dim=1)
          memory_mask = 1 - torch.eq(labels, memory_labels.T).float().to(device)
          contrast_feature = torch.cat([anchor_feature, memory_features]).detach()
          #self_contrast_mask = 1 - torch.diag(torch.ones((mask.size()[0])))
          #logits_mask = torch.cat((self_contrast_mask.to(device), memory_mask.to(device)),dim=1)
          logits_mask = torch.ones_like(mask).to(device)
          self_contrast_mask = 1 - torch.diag(torch.ones((mask.size()[0])))
          logits_mask[:,:mask.size()[0]] = logits_mask[:,:mask.size()[0]].clone() * self_contrast_mask.to(device)
          #exit()


        # compute logits
        anchor_norm = torch.norm(anchor_feature,dim=1)
        contrast_norm = torch.norm(contrast_feature,dim=1)
        anchor_feature = anchor_feature/(anchor_norm.unsqueeze(1))
        contrast_feature = contrast_feature/(contrast_norm.unsqueeze(1))
        # print(anchor_feature.sum())
        # print('%'*400)
        # print(contrast_feature.sum())
        # print('%'*40)
        mul = torch.matmul(anchor_feature, contrast_feature.T)
        anchor_dot_contrast =  1 - torch.div(1,1+torch.exp(mul))
        # anchor_dot_contrast = torch.div(mul,0.07)
        if weighted:
          lenx = anchor_dot_contrast.shape[0]
          widx = anchor_dot_contrast.shape[1]
          weights = torch.tensor([[(lenx - np.abs(i-j))/lenx for i in range(1,lenx+1)] for j in range(1,lenx+1)]).to(device)
          diff = widx - lenx
          if diff != 0:
            mb_weights = torch.empty(lenx, diff).fill_(1/lenx).to(device)
            weights = torch.cat((weights, mb_weights),dim=1)
          weights = weights * mask
          weights = torch.where(weights > 0,weights, 1)
          # print(anchor_dot_contrast.shape)


          # print(weights)
          anchor_dot_contrast = anchor_dot_contrast * weights
        # torch.matmul(anchor_norm, contrast_norm.T)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        #logits = anchor_dot_contrast
        # tile mask

        '''
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange( contrast_count * anchor_count).view(-1, 1).to(device),
            0
        )
        '''
        mask = mask * logits_mask
        nonzero_index = torch.where(mask.sum(1)!=0)[0]
        if len(nonzero_index) == 0:
          print('here')
          print('*'*40)
          return torch.tensor([0]).float().to(device)
        # compute log_prob
        mask = mask[nonzero_index]
        logits_mask = logits_mask[nonzero_index]
        logits = logits[nonzero_index]
        exp_logits = torch.exp(logits) * logits_mask
        # exp_logits = logits * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # log_prob = logits/exp_logits.sum(1, keepdim=True)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - mean_log_prob_pos
        # print("loss:",loss)
        # print('*'*40)
        loss = loss.nanmean()
        return loss


class CRFOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''
    def __init__(self, in_dim, num_labels):
        super(CRFOutputLayer, self).__init__()
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(in_dim, self.num_labels)
        self.crf = ConditionalRandomField(self.num_labels)

    def forward(self, x, mask, labels=None):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''

        batch_size, max_sequence, in_dim = x.shape

        logits = self.classifier(x)
        outputs = {}
        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask)
            loss = -log_likelihood
            outputs["loss"] = loss
        else:
            best_paths = self.crf.viterbi_tags(logits, mask)
            predicted_label = [x for x, y in best_paths]
            predicted_label = [pad_sequence_to_length(x, desired_length=max_sequence) for x in predicted_label]
            predicted_label = torch.tensor(predicted_label)
            outputs["predicted_label"] = predicted_label
            outputs["logits"] = logits
        return outputs




class AttentionPooling(torch.nn.Module):
    def __init__(self, in_features, dimension_context_vector_u=200, number_context_vectors=5):
        super(AttentionPooling, self).__init__()
        self.dimension_context_vector_u = dimension_context_vector_u
        self.number_context_vectors = number_context_vectors
        self.linear1 = torch.nn.Linear(in_features=in_features, out_features=self.dimension_context_vector_u, bias=True)
        self.linear2 = torch.nn.Linear(in_features=self.dimension_context_vector_u,
                                       out_features=self.number_context_vectors, bias=False)

        self.output_dim = self.number_context_vectors * in_features

    def forward(self, tokens, mask):
        #shape tokens: (batch_size, tokens, in_features)

        # compute the weights
        # shape tokens: (batch_size, tokens, dimension_context_vector_u)
        a = self.linear1(tokens)
        a = torch.tanh(a)
        # shape (batch_size, tokens, number_context_vectors)
        a = self.linear2(a)
        # shape (batch_size, number_context_vectors, tokens)
        a = a.transpose(1, 2)
        a = masked_softmax(a, mask)

        # calculate weighted sum
        s = torch.bmm(a, tokens)
        s = s.view(tokens.shape[0], -1)
        return s



class BertTokenEmbedder(torch.nn.Module):
    def __init__(self, config):
        super(BertTokenEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_model"])
        self.bert_trainable = config["bert_trainable"]
        self.bert_hidden_size = self.bert.config.hidden_size
        self.cacheable_tasks = config["cacheable_tasks"]
        for param in self.bert.parameters():
            param.requires_grad = self.bert_trainable

    def forward(self, batch):
        documents, sentences, tokens = batch["input_ids"].shape

        if "bert_embeddings" in batch:
            return batch["bert_embeddings"]

        attention_mask = batch["attention_mask"].view(-1, tokens)
        input_ids = batch["input_ids"].view(-1, tokens)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # shape (documents*sentences, tokens, 768)
        bert_embeddings = outputs[0]


        if not self.bert_trainable:
            # cache the embeddings of BERT if it is not fine-tuned
            # to save GPU memory put the values on CPU
            batch["bert_embeddings"] = bert_embeddings.to("cpu")

        return bert_embeddings

class BertHSLN(torch.nn.Module):
    '''
    Model for Baseline, Sequential Transfer Learning and Multitask-Learning with all layers shared (except output layer).
    '''
    def __init__(self, config, num_labels):
        super(BertHSLN, self).__init__()
        self.use_crf = config['use_crf']
        self.num_labels = num_labels
        self.bert = BertTokenEmbedder(config)
        self.sl = config['supervised_loss']
        self.mb = config['memory_bank']
        self.sp = config['single_proto']
        try:
          self.mp = config['multi_proto']
        except:
          self.mp = False 
        if self.sl:
          self.contrastive_loss = SupConLoss()
          self.weighted = config['weighted']
          
          self.memory_bank = None
          self.memory_bank_size = config['memory_bank_size']
          
          self.min_class_label = math.floor(config['memory_bank_size']/config['nlabels'])
          self.ld = dict.fromkeys(np.arange(0,config['nlabels']))

        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])

        self.lstm_hidden_size = config["word_lstm_hs"]
        if self.sp:
          self.proto_sim_model = ProtoSimModel(self.num_labels, self.lstm_hidden_size * 2)
        if self.mp:
          self.proto_sim_model = MultiProtoSimModel(self.num_labels, self.lstm_hidden_size * 2)

        self.classifier = torch.nn.Linear(self.lstm_hidden_size * 2, self.num_labels) 

        self.word_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=self.bert.bert_hidden_size,
                                  hidden_size=self.lstm_hidden_size,
                                  num_layers=1, batch_first=True, bidirectional=True))

        self.attention_pooling = AttentionPooling(2 * self.lstm_hidden_size,
                                                  dimension_context_vector_u=config["att_pooling_dim_ctx"],
                                                  number_context_vectors=config["att_pooling_num_ctx"])

        self.init_sentence_enriching(config)
        self.reinit_output_layer(config)

    def enqueue_and_dequeue(self, batch_examples):
      if self.memory_bank==None:
        self.memory_bank = batch_examples
      else:
        self.memory_bank = torch.cat([self.memory_bank, batch_examples.detach()] , dim=0)
        if self.memory_bank.size()[0] > self.memory_bank_size:
          temp1 = []
          for k in self.ld.keys():
            temp3 = [x for x in self.memory_bank if x[0]==k][-self.min_class_label:]
            if temp3:
              temp1.append(torch.stack(temp3, dim=0))    
          self.memory_bank = torch.cat(temp1,dim=0)


    def init_sentence_enriching(self, config):
        input_dim = self.attention_pooling.output_dim
        print(f"Attention pooling dim: {input_dim}")
        self.sentence_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=input_dim,
                                  hidden_size=self.lstm_hidden_size,
                                  num_layers=1, batch_first=True, bidirectional=True))


    def reinit_output_layer(self, config):
        input_dim = self.lstm_hidden_size * 2
        self.crf = CRFOutputLayer(in_dim=input_dim, num_labels=self.num_labels)
        
        
    def forward(self, batch, labels=None, get_embeddings = False):

        documents, sentences, tokens = batch["input_ids"].shape

        # shape (documents*sentences, tokens, 768)
        bert_embeddings = self.bert(batch)

        # in Jin et al. only here dropout
        bert_embeddings = self.dropout(bert_embeddings)

        tokens_mask = batch["attention_mask"].view(-1, tokens)
        # shape (documents*sentences, tokens, 2*lstm_hidden_size)
        bert_embeddings_encoded = self.word_lstm(bert_embeddings, tokens_mask)

        # shape (documents*sentences, pooling_out)
        # sentence_embeddings = torch.mean(bert_embeddings_encoded, dim=1)
        sentence_embeddings = self.attention_pooling(bert_embeddings_encoded, tokens_mask)
        # shape: (documents, sentences, pooling_out)
        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
        # in Jin et al. only here dropout
        sentence_embeddings = self.dropout(sentence_embeddings)

        
        sentence_mask = batch["sentence_mask"]

        # shape: (documents, sentence, 2*lstm_hidden_size)
        sentence_embeddings_encoded = self.sentence_lstm(sentence_embeddings, sentence_mask)
        # in Jin et al. only here dropout
        sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded)
        sentence_embeddings_encoded = sentence_embeddings_encoded.squeeze()
        # sentence_embeddings_encoded_nodrop = sentence_embeddings_encoded_nodrop.squeeze()

        logits = self.classifier(sentence_embeddings_encoded)
        if self.use_crf:
          output = self.crf(sentence_embeddings_encoded, sentence_mask, labels)
        else:
          output = {}
          if labels is not None:
            logits = logits.squeeze()
            labels = labels.squeeze()
            predicted_labels = torch.argmax(logits, dim=1)
            output['predicted_label'] = predicted_labels

            loss = F.cross_entropy(logits, labels)
            if self.sp:
              pc_loss = self.proto_sim_model.get_proto_centric_loss(sentence_embeddings_encoded, labels)
              sc_loss = self.proto_sim_model.get_sample_centric_loss(sentence_embeddings_encoded, labels)
              cls_loss = self.proto_sim_model.get_classification_loss(sentence_embeddings_encoded, labels)
              output['pc_loss'] = pc_loss
              output['sc_loss'] = sc_loss
              output['cls_loss'] = cls_loss
            elif self.mp:
              pc_loss = self.proto_sim_model.get_proto_centric_loss(sentence_embeddings_encoded, labels)
              sc_loss = self.proto_sim_model.get_sample_centric_loss(sentence_embeddings_encoded, labels)
              cls_loss = self.proto_sim_model.get_classification_loss(sentence_embeddings_encoded, labels)
            
              output['loss'] = loss
              output['sc_loss'] = sc_loss
              output['cls_loss'] = cls_loss


            output['loss'] = loss
            output['logits']=logits
          else:
            logits = logits.squeeze()
            predicted_labels = torch.argmax(logits, dim=1)
            output['predicted_label'] = predicted_labels
            output['logits']=logits

        if self.sl and labels != None:
          if self.memory_bank == None or self.mb == False:
            c_loss = self.contrastive_loss(sentence_embeddings_encoded, torch.squeeze(labels), weighted = self.weighted)
          else:
            memory_label = self.memory_bank[:,:1].squeeze().detach()
            memory_feature = self.memory_bank[:, 1:].detach()
            c_loss = self.contrastive_loss(sentence_embeddings_encoded,  torch.squeeze(labels),
              memory_features= memory_feature, memory_labels=memory_label, weighted = self.weighted)
          if self.mb:
            batch_example = torch.cat([torch.squeeze(labels).unsqueeze(1), sentence_embeddings_encoded], dim = 1 ).detach()
            self.enqueue_and_dequeue(batch_example)  
            

          output['cont_loss'] = c_loss 
        if get_embeddings:
          return output, sentence_embeddings_encoded
        
        return output
