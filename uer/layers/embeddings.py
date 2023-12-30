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

    def forward(self, src, seg, type, pos):
        word_emb = self.word_embedding(src)
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

