import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv

class HGTModel(nn.Module):
    def __init__(self, data, hidden_dim=64, num_layers=2, num_heads=2):
        super().__init__()

        self.node_types = data.node_types
        self.edge_types = data.edge_types

        self.hidden_dim = hidden_dim

        # -------------------------
        # 1. Node-type embeddings
        # -------------------------
        self.node_emb = nn.ModuleDict({
            ntype: nn.Embedding(data[ntype].num_nodes, hidden_dim)
            for ntype in self.node_types
        })

        # -------------------------
        # 2. HGT layers
        # -------------------------
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=(self.node_types, self.edge_types),
                    heads=num_heads
                )
            )

        # -------------------------
        # 3. Normalization
        # -------------------------
        self.norms = nn.ModuleDict({
            ntype: nn.LayerNorm(hidden_dim)
            for ntype in self.node_types
        })

        self.activation = nn.ReLU()


    def forward(self, data):
        x_dict = {}

        print(data.metadata)

        

        # -------------------------
        # Step 1: initialize embeddings
        # -------------------------
        for ntype in data.node_types:
            num_nodes = data[ntype].num_nodes

            node_ids = torch.arange(
                num_nodes,
                device=next(self.parameters()).device
            )

            x = self.node_emb[ntype](node_ids)
            x = self.norms[ntype](x)
            x_dict[ntype] = x

        # -------------------------
        # Step 2: HGT layers
        # -------------------------
        for conv in self.layers:
            x_prev = x_dict

            x_dict = conv(
                x_dict,
                data.edge_index_dict
            )

            # activation + residual + norm
            for ntype in x_dict:
                x_dict[ntype] = self.activation(x_dict[ntype])
                x_dict[ntype] = x_dict[ntype] + x_prev[ntype]
                x_dict[ntype] = self.norms[ntype](x_dict[ntype])

        return x_dict