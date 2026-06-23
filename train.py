import copy
import torch
import random

from models import *

from torch_geometric.loader import LinkNeighborLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def evaluate(model,eval_loader, predictor, tgt_edge_type, device):
    model.eval()
    predictor.eval()
    all_scores = []
    all_labels = []
    for batch in eval_loader:
        batch = batch.to(device)
        z_dict = model(batch)
        src, dst = batch[tgt_edge_type]['edge_label_index']

        gene_emb = z_dict['biolink:Gene'][dst]
        disease_emb = z_dict['biolink:Disease'][src]

        scores = predictor(gene_emb, disease_emb).sigmoid()
        labels = batch[tgt_edge_type]['edge_label']

        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())
    all_scores = torch.cat(all_scores, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    
    return roc_auc_score(all_labels, all_scores)



def train(model,train_loader, predictor, optimizer, tgt_edge_type, device):
    model.train()
    predictor.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        z_dict = model(batch)
        src, dst = batch[tgt_edge_type]['edge_label_index']

        gene_emb = z_dict['biolink:Gene'][dst]
        disease_emb = z_dict['biolink:Disease'][src]

        pred = predictor(gene_emb, disease_emb)

        edge = (pred > 0).int()

        loss = F.binary_cross_entropy_with_logits(pred,batch[tgt_edge_type]['edge_label'].float())
        total_loss = total_loss + loss
        loss.backward()
        optimizer.step()
        #print(edge, type(edge))
        #print(batch[etype]['edge_label'], type(batch[etype]['edge_label']))
        #correct = (edge == batch[tgt_edge_type]['edge_label']).float()
        #accuracy = correct.mean()
        #print("Accuracy: ", accuracy)

    return total_loss / len(train_loader)



def mask_edges(data, mask_ratio=0.1):
    """
    Masks edges per relation type (IMPORTANT for KG balance)
    """

    train_data = copy.deepcopy(data)
    pos_edges = {}

    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index

        num_edges = edge_index.size(1)
        perm = torch.randperm(num_edges)

        mask_size = int(mask_ratio * num_edges)

        mask_idx = perm[:mask_size]
        keep_idx = perm[mask_size:]

        # store positives for loss
        pos_edges[edge_type] = edge_index[:, mask_idx]

        # keep remaining edges for message passing
        train_data[edge_type].edge_index = edge_index[:, keep_idx]

    return train_data, pos_edges

def negative_sampling(edge_index, num_nodes, num_neg=1):
    """
    Replace tail nodes randomly
    """
    heads = edge_index[0]
    tails = edge_index[1]

    neg_tails = torch.randint(0, num_nodes, (tails.size(0) * num_neg,))

    neg_heads = heads.repeat(num_neg)

    neg_edges = torch.stack([neg_heads, neg_tails], dim=0)

    return neg_edges

