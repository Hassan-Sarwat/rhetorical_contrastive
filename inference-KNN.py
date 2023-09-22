def get_logits(logits, embeddings, datastore_keys, datastore_values, num_labels,K, lambda_, link_temperature=1.0):
  # cosine similarity
  knn_feats = datastore_keys.squeeze(0).transpose(0, 1) # [feature_size=768, datastore_size]
  embeddings = embeddings.view(-1, embeddings.shape[-1])  # [sentences, feature_size=768]
  sim = torch.mm(embeddings, knn_feats) # [sentences, datastore_size]

  sentences = embeddings.shape[0]
  datastore_size = knn_feats.shape[1]

  norm_1 = (knn_feats ** 2).sum(dim=0, keepdim=True).sqrt() # [1, datastore_size]
  norm_2 = (embeddings ** 2).sum(dim=1, keepdim=True).sqrt() # [sentences, 1]
  scores = (sim / (norm_1 + 1e-10) / (norm_2 + 1e-10)).view(1, sentences, -1) # [1, sentences, datastore_size]
  knn_labels = datastore_values.view(1, 1, datastore_size).expand(1, sentences, datastore_size) # [1, sentences, datastore_size]

  # select scores and labels of the top k only
  topk_scores, topk_idxs = torch.topk(scores, dim=-1, k=K)  # [1, sentences, topk]
  scores = topk_scores
  knn_labels = knn_labels.gather(dim=-1, index=topk_idxs)  # [[1, sentences, topk]

  # transform scores to softmax probabilities
  sim_probs = torch.softmax(scores / link_temperature, dim=-1) # [[1, sentences, topk]

  # 1. create zero tensor for probabilites as placeholder
  knn_probabilities = torch.zeros_like(sim_probs[:, :, 0]).unsqueeze(-1).repeat([1, 1, num_labels])  # [1, sentences, num_labels]
  # for each row (dim=2)
  # sum the probabilities from sim softmax probabilities (src=sim_probs) grouped by class (index=knn_labels)
  knn_probabilities = knn_probabilities.scatter_add(dim=2, index=knn_labels, src=sim_probs) # [1, sentences, num_labels]

  # interpolate between logits and knn_probabilites
  probabilities = lambda_*logits + (1-lambda_)*knn_probabilities

  # argmax to get most likely label
  argmax_labels = torch.argmax(probabilities, 2, keepdim=False)

  # return predicted labels
  return argmax_labels




f1s = {}
for lambda_ in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    f1s[lambda_] = {}
    for k in [5,10,24,124,256,512]:
        predicted_labels_protoype_average = get_logits(logits, embeddings, datastore_keys, datastore_values, 7, k, lambda_)
        f1 = get_f1_score(labels, predicted_labels_protoype_average, average='macro')
        f1s[lambda_][k] = f1