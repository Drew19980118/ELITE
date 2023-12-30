# -*- encoding:utf-8 -*-
import torch.nn as nn
import torch
from torch.nn import GELU
from uer.layers.transformer import TransformerLayer
from uer.layers.GAT import GATLayer
import torch.nn.functional as F


class BertEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.transformer = nn.ModuleList([
            TransformerLayer(args) for _ in range(self.layers_num)
        ])
        self.activation = GELU()
        self.gnn_layers = nn.ModuleList(
            [GATLayer(768, 4) for _ in range(5)])

    def mp_helper(self, _X, edge_index):
        for _ in range(5):
            _X = self.gnn_layers[_](_X, edge_index)
            _X = self.activation(_X)
            _X = F.dropout(_X, 0.1, training = self.training)
        return _X
        
    def forward(self, ids, concept_ent_pairs, edge_idx, emb, seg, need_gnn, vm=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
            vm: [batch_size x seq_length x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        seq_length = emb.size(1)
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        if vm is None:
            mask = (seg > 0). \
                    unsqueeze(1). \
                    repeat(1, seq_length, 1). \
                    unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0
        else:
            mask = vm.unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0

        hidden = emb
        for i in range(self.layers_num):
            hidden = self.transformer[i](hidden, mask)
        if need_gnn == True:
            gnn_emb_batch = torch.empty(0, 768)
            gnn_text_node_idx = []
            word_gnn_idx = []
            #not_word_gnn_idx = []
            for index, concept_ent_pair in enumerate(concept_ent_pairs):
                if len(concept_ent_pair) > 0:
                    word_gnn_idx.append(index)
                    edge_idx[index] = edge_idx[index] + gnn_emb_batch.size(0)
                    gnn_emb_batch = torch.cat((gnn_emb_batch, hidden[index, 0, :].unsqueeze(0)), dim=0)
                    # gnn_emb_batch = torch.cat((gnn_emb_batch, text_node_batch[index].unsqueeze(0)), dim=0)
                    gnn_text_node_idx.append(gnn_emb_batch.size(0) - 1)
                    for concept_ent in concept_ent_pair:
                        start_seq = concept_ent[0]
                        end_seq = concept_ent[-1]
                        gnn_emb_batch = torch.cat((gnn_emb_batch, ((hidden[index][start_seq:end_seq + 1].sum(0))).unsqueeze(0)), dim=0)

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
                    hidden[idx][0] = select_output[index].squeeze(0)
        return hidden
