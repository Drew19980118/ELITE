import torch
from torch import nn
from torch_geometric.utils import softmax
from torch_scatter import scatter
import math

class GATLayer(nn.Module):

    def __init__(self, emb_dim, head_count=4):
        super(GATLayer, self).__init__()

        # For attention
        self.head_count = head_count
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(emb_dim, head_count * self.dim_per_head)

        self.aggr = "add"

        self._alpha = None

        # For final MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))

    def forward(self, x, edge_index):
        x_i, x_j = self.collect(x, edge_index)
        out = self.message(edge_index, x_i, x_j)
        out = self.aggregate(out, edge_index[1], x.size(0), self.aggr)
        out = self.mlp(out)
        self._alpha = None
        return out


    def collect(self, x, edge_index):
        X = x[edge_index]
        return X[1], X[0]

    def message(self, edge_index, x_i, x_j):
        key = self.linear_key(x_i).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]
        msg = self.linear_msg(x_j).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]

        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key)
        scores = scores.sum(dim=2)  # [E, heads]
        src_node_index = edge_index[0]  # [E,]
        alpha = softmax(scores, src_node_index)  # [E, heads] #group by src side node
        self._alpha = alpha

        # adjust by outgoing degree of src
        E = edge_index.size(1)  # n_edges
        N = int(src_node_index.max()) + 1  # n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index]  # [E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1)  # [E, heads]

        out = msg * alpha.view(-1, self.head_count, 1)  # [E, heads, _dim]
        return out.view(-1, self.head_count * self.dim_per_head)  # [E, emb_dim]

    def aggregate(self, inputs, index, dim_size, reduce):
        return scatter(inputs, index, dim=-2, dim_size=dim_size, reduce=reduce)
