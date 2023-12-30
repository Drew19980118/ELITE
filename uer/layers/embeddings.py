# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import GELU

from uer.layers.layer_norm import LayerNorm
from uer.layers.GAT import GATLayer
import torch.nn.functional as F
import h5py


class BertEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(BertEmbedding, self).__init__()
        self.emb_size = args.emb_size
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.pad_type_embedding = torch.zeros(args.emb_size)
        self.token_type_embeddings = nn.Embedding(21, args.emb_size, _weight=torch.zeros((21, args.emb_size)))
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)
        self.activation = GELU()
        self.dropout = nn.Dropout(0.1)
        self.gnn_layers = nn.ModuleList(
            [GATLayer(768, 4) for _ in range(5)])

    def mp_helper(self, _X, edge_index):
        for _ in range(5):
            _X = self.gnn_layers[_](_X, edge_index)
            _X = self.activation(_X)
            _X = F.dropout(_X, 0.1, training = self.training)
        return _X

    def forward(self, ids, src, seg, type, concept_ent_pairs, edge_idx, pos, need_gnn):
        word_emb = self.word_embedding(src)
        if need_gnn == True:
            gnn_emb_batch = torch.empty(0, 768)
            gnn_text_node_idx = []
            word_gnn_idx = []
            for index, concept_ent_pair in enumerate(concept_ent_pairs):
                if len(concept_ent_pair) > 0:
                    word_gnn_idx.append(index)
                    edge_idx[index] = edge_idx[index] + gnn_emb_batch.size(0)
                    gnn_emb_batch = torch.cat((gnn_emb_batch, word_emb[index, 0, :].unsqueeze(0)), dim=0)
                    gnn_text_node_idx.append(gnn_emb_batch.size(0) - 1)
                    for concept_ent in concept_ent_pair:
                        start_seq = concept_ent[0]
                        end_seq = concept_ent[-1]
                        gnn_emb_batch = torch.cat((gnn_emb_batch, ((word_emb[index][start_seq:end_seq + 1].sum(0))).unsqueeze(0)), dim=0)

            edge_idx = [edge for edge in edge_idx if edge.shape[1] != 0]
            if len(edge_idx) != 0:
                edge_idx = torch.cat(edge_idx, dim=1)
                new_edge_idx = torch.zeros(edge_idx.shape, dtype=torch.long)
                new_edge_idx[0] = edge_idx[1]
                new_edge_idx[1] = edge_idx[0]
                new_edge_idx = torch.cat((edge_idx, new_edge_idx), dim=1)
            if gnn_emb_batch.shape[0] != 0:
                output = self.mp_helper(gnn_emb_batch, new_edge_idx)
                output = self.dropout(output)
                select_output = output[gnn_text_node_idx]
                for index, idx in enumerate(word_gnn_idx):
                    word_emb[idx][0] = select_output[index].squeeze(0)
        if pos is None:
            pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, \
                                        dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        else:
            pos_emb = self.position_embedding(pos)
        seg_emb = self.segment_embedding(seg)
        type_emb = self.token_type_embeddings(type)
        emb = word_emb + pos_emb + seg_emb + type_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb

