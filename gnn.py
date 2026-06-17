from collections import defaultdict
import pandas as pd
from torch_geometric.data import HeteroData
import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import to_hetero
from itertools import islice



KG_NODES_FILE = '/hpc/users/sharmh15/labhome/projects/genediseaseassociation/monarch/monarchkg/monarch-kg_nodes.tsv'
KG_EDGES_FILE = '/hpc/users/sharmh15/labhome/projects/genediseaseassociation/monarch/monarchkg/monarch-kg_edges.tsv'
RELEVANT_NODE_FIELDS = ['id', 'category','name', 'type', 'description' , 'full_name' , 'in_taxon',  'in_taxon_label',  'symbol',  'synonym', 
     'exact_synonym', 'broad_synonym', 'narrow_synonym',   'related_synonym',  'deprecated' , 'subsets',  'has_gene', 'has_biological_sex']
RELEVANT_EDGE_FIELDS = ['id', 'predicate', 'category', 'agent_type', 'knowledge_level',
                        'negated',
                        'has_evidence', 
                        'onset_qualifier',  'frequency_qualifier',  'sex_qualifier', 'stage_qualifier', 'species_context_qualifier' ,
                        'disease_context_qualifier',
                        'has_count', 'has_percentage', 'has_quotient', 'has_total',
                        'subject', 'object'
                        ]

# Maybe I should pass list of Nodes
def create_hetero_graph_from_monarch():
    graph = HeteroData()
    df = pd.read_csv(KG_NODES_FILE,sep='\t')
    df = df[RELEVANT_NODE_FIELDS]
    categorywisedf = {cat: group for cat, group in df.groupby('category')}
    id_to_category_idces = defaultdict(list)
    id_to_category = {}
    for k,v in categorywisedf.items():
        id_to_category_idces[k] = {id:ix for (ix,id) in enumerate(v['id'])} 
        id_to_category = {id:k for id in v['id']}
        graph[k].num_nodes = v.shape[0]
        graph[k].name = v['name'].tolist()
        graph[k].type = v['type'].tolist()
        graph[k].description = v['description'].tolist()
        graph[k].full_name = v['full_name'].tolist()
        graph[k].in_taxon = v['in_taxon'].tolist()
        graph[k].in_taxon_label = v['in_taxon_label'].tolist()
        graph[k].symbol = v['symbol'].tolist()
        graph[k].synonym = v['synonym'].tolist()
        graph[k].exact_synonym = v['exact_synonym'].tolist()
        graph[k].broad_synonym = v['broad_synonym'].tolist()
        graph[k].narrow_synonym = v['narrow_synonym'].tolist()
        graph[k].description = v['description'].tolist()
        graph[k].related_synonym = v['related_synonym'].tolist()
        graph[k].deprecated = v['deprecated'].tolist()
        graph[k].subsets = v['subsets'].tolist()
        graph[k].has_gene = v['has_gene'].tolist()
        graph[k].has_biological_sex = v['has_biological_sex'].tolist()

    # Nodes have been added. 
    df = pd.read_csv(KG_EDGES_FILE,sep='\t')
    df = df[RELEVANT_EDGE_FIELDS]

    edge_groups = defaultdict(list)

    df['object_category'] = [id_to_category[e] for e in df['object']]
    df['subject_category'] = [id_to_category[e] for e in df['subject']]

    pd.set_option('display.max_rows', 100)

    print(df.head(30))
    print(id_to_category)
    


    
    #for k,v in id_to_category_idces.items():
    #    few_elements = dict(islice(v.items(), 30))
    #    print(k, few_elements)

    

    print(graph.metadata)
    

    return graph



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
        relation = relation.replace('|', '_')
        
        if relation == "part_of":
            invrelation = "has_part"
        else:
            invrelation = 'inv_' + relation

        graph["gene",relation,"go"].edge_index = torch.tensor([src, tgt],dtype=torch.long)
        graph["go", invrelation, "gene"].edge_index = torch.tensor([tgt, src],dtype=torch.long)
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

    
    


