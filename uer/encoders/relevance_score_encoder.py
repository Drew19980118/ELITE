import torch.nn as nn
from uer.layers.relevance_transformer import RelevanceTransformerLayer
class RelevanceEncoder(nn.Module):
        """
        BERT encoder exploits 12 or 24 transformer layers to extract features.
        """
        def __init__(self, args):
            super(RelevanceEncoder, self).__init__()
            self.layers_num = args.layers_num
            self.transformer = nn.ModuleList([
                RelevanceTransformerLayer(args) for _ in range(self.layers_num)
            ])

        def forward(self, emb, seg):
            """
            Args:
                emb: [batch_size x seq_length x emb_size]
                seg: [batch_size x seq_length]
                vm: [batch_size x seq_length x seq_length]

            Returns:
                hidden: [batch_size x seq_length x hidden_size]
            """
            mask = seg.float()
            mask = (1.0 - mask) * -10000.0
            hidden = emb
            for i in range(self.layers_num):
                hidden = self.transformer[i](hidden, mask)
            return hidden