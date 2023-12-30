# -*- encoding:utf-8 -*-
from uer.utils.subword import *

class RelevanceScoreModel(nn.Module):
    """
    BertModel consists of three parts:
        - embedding: token embedding, position embedding, segment embedding
        - encoder: multi-layer transformer encoders
        - target: mlm and nsp tasks
    """
    def __init__(self, args, embedding, encoder):
        super(RelevanceScoreModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder

    def forward(self, src):
        # [batch_size, seq_length, emb_size]
        emb = self.embedding(src)
        output = self.encoder(emb)
        return output
