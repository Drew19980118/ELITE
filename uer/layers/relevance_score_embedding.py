# -*- encoding:utf-8 -*-
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm

class RelevanceScoreEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(RelevanceScoreEmbedding, self).__init__()
        self.emb_size = args.emb_size
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src):
        word_emb = self.word_embedding(src)
        # device = torch.device("cuda:0")  # 定义目标设备，这里假设是第一个GPU
        # type_emb = type_emb.to(device)
        emb = self.dropout(self.layer_norm(word_emb))
        return emb

