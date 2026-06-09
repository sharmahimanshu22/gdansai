from collections import defaultdict
from torch_geometric.data import HeteroData
import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import to_hetero



def create_hetero_graph(go_nodes, gene_nodes):
    graph = HeteroData()
    graph["go"].num_nodes = len(go_nodes)
    graph["gene"].num_nodes = len(gene_nodes)

    # quite useless. we will see
    graph["go"].metadata = None
    graph["gene"].metadata = None

    go_nodes = dict(sorted(go_nodes.items()))
    gene_nodes = dict(sorted(gene_nodes.items()))

    go_to_idx = {k:i for i,(k,v) in enumerate(go_nodes.items()) }
    gene_to_idx = {k:i for i,(k,v) in enumerate(gene_nodes.items()) }

    go_to_go_edge_gropus = defaultdict(list)
    gene_to_go_edge_gropus = defaultdict(list)
    
    for (key, value) in go_nodes.items():
        src_idx = go_to_idx[key]
        neighbors = value['neighbors']
        for e in neighbors:
            target = e[0]
            if target == 'Thing':
                continue
            target_idx = go_to_idx[target]
            relation = e[1]
            go_to_go_edge_gropus[relation].append((src_idx, target_idx))
    
    for (key, value) in gene_nodes.items():
        src_idx = gene_to_idx[key]
        neighbors = value['neighbors']
        for e in neighbors:
            target = e[0]
            target_idx = go_to_idx[target]
            relation = e[1]
            gene_to_go_edge_gropus[relation].append((src_idx, target_idx))

    
    for relation,edges in go_to_go_edge_gropus.items():
        src = [e[0] for e in edges]
        tgt = [e[1] for e in edges]
        graph["go",relation,"go"].edge_index = torch.tensor([src, tgt],dtype=torch.long)
        print(graph["go",relation,"go"].edge_index)
        #print(torch.tensor([src, tgt],dtype=torch.long))
    
    for relation,edges in gene_to_go_edge_gropus.items():
        src = [e[0] for e in edges]
        tgt = [e[1] for e in edges]
        graph["gene",relation,"go"].edge_index = torch.tensor([src, tgt],dtype=torch.long)
        print(graph["gene",relation,"go"].edge_index)

        #print(torch.tensor([src, tgt],dtype=torch.long))

    return graph


import torch
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv



class NodeEncoder(torch.nn.Module):

    def __init__(self, data, hidden_dim):

        super().__init__()

        self.go_emb = torch.nn.Embedding(
            data["go"].num_nodes,
            hidden_dim
        )

        self.gene_emb = torch.nn.Embedding(
            data["gene"].num_nodes,
            hidden_dim
        )

    def forward(self):

        return {
            "go": self.go_emb.weight,
            "gene": self.gene_emb.weight
        }
    
class GNN(torch.nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.conv1 = SAGEConv(
            (-1, -1),
            hidden_dim
        )

        self.conv2 = SAGEConv(
            (-1, -1),
            hidden_dim
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)

        return x
    

class OntologyModel(torch.nn.Module):

    def __init__(self, data):

        super().__init__()

        self.encoder = NodeEncoder(
            data,
            hidden_dim=256
        )

        self.gnn = to_hetero(
            GNN(256),
            data.metadata()
        )

    def forward(self, data):

        x_dict = self.encoder()

        x_dict = self.gnn(
            x_dict,
            data.edge_index_dict
        )

        return x_dict
    

class OntologyRGCN(torch.nn.Module):

    def __init__(
        self,
        num_nodes,
        num_relations,
        hidden_dim=256
    ):
        super().__init__()

        self.embedding = torch.nn.Embedding(
            num_nodes,
            hidden_dim
        )

        self.conv1 = RGCNConv(
            hidden_dim,
            hidden_dim,
            num_relations
        )

        self.conv2 = RGCNConv(
            hidden_dim,
            hidden_dim,
            num_relations
        )

    def forward(
        self,
        edge_index,
        edge_type
    ):

        x = self.embedding.weight

        x = self.conv1(
            x,
            edge_index,
            edge_type
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index,
            edge_type
        )

        return x

    

#model = OntologyRGCN(
##    num_nodes=homo_data.num_nodes,
#    num_relations=len(data.edge_types)
#)


    
    #print(go_to_go_edge_gropus)
    #print(gene_to_go_edge_gropus)
    #graph["go",relation,"go"].edge_index = torch.tensor([go_to_idx(key), go_to_idx(target)],dtype=torch.long)
    #graph["gene",relation,"go"].edge_index = torch.tensor([gene_to_idx(key), go_to_idx(target)],dtype=torch.long)

    
    


