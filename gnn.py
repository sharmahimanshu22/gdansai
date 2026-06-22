from collections import defaultdict
import pandas as pd
from torch_geometric.data import HeteroData
import torch
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.nn import to_hetero
from itertools import islice
import random
import torch.nn as nn

import torch
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv
import sys
import gc



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
        id_to_category_current = {id:k for id in v['id']}
        id_to_category.update(id_to_category_current)
        
        graph[k].num_nodes = v.shape[0]
        #graph[k].name = v['name'].tolist()
        #graph[k].type = v['type'].tolist()
        #graph[k].description = v['description'].tolist()
        #graph[k].full_name = v['full_name'].tolist()
        #graph[k].in_taxon = v['in_taxon'].tolist()
        #graph[k].in_taxon_label = v['in_taxon_label'].tolist()
        #graph[k].symbol = v['symbol'].tolist()
        #graph[k].synonym = v['synonym'].tolist()
        #graph[k].exact_synonym = v['exact_synonym'].tolist()
        #graph[k].broad_synonym = v['broad_synonym'].tolist()
        #graph[k].narrow_synonym = v['narrow_synonym'].tolist()
        #graph[k].description = v['description'].tolist()
        #graph[k].related_synonym = v['related_synonym'].tolist()
        #graph[k].deprecated = v['deprecated'].tolist()
        #graph[k].subsets = v['subsets'].tolist()
        #graph[k].has_gene = v['has_gene'].tolist()
        #graph[k].has_biological_sex = v['has_biological_sex'].tolist()
        

        #Random Initialization of Embeddings without using any features
        #graph[k].x = nn.Embedding(graph[k].num_nodes,128)

    del df
    gc.collect()

    #random_items = random.sample(list(id_to_category.items()), 100)
    #print(random_items)


    # Nodes have been added. Now read the edges file and add reverse edges
    df = pd.read_csv(KG_EDGES_FILE,sep='\t')
    df = df[RELEVANT_EDGE_FIELDS]
    df['object_category'] = [id_to_category[e] for e in df['object']]
    df['subject_category'] = [id_to_category[e] for e in df['subject']]
    dfs = {name: group for name, group in df.groupby(['object_category', 'predicate' ,'subject_category'])}
    dfsorig = {name: group for name, group in df.groupby(['object_category', 'predicate' ,'subject_category'])}
    dfsnew = {}
    for k in dfs:
        inverse_predicate = get_biological_inverse_if_exists(k[1])
        #print("\n",inverse_predicate)
        if inverse_predicate is not None:
            df1 = dfs.get(k)
            if k[0] == k[2]:
                #print(k , " same src target")
                df2 = df1.copy().rename(columns={'object': 'subject', 'subject': 'object', 
                                        'object_category' : 'subject_category', 'subject_category' : 'object_category' })
                df1 = pd.concat([df1,df2], ignore_index=True)
                df1 = df1.drop_duplicates()
                dfs[k] = df1
            else:
                rev_key = (k[2], inverse_predicate, k[0])
                if rev_key in dfs:
                    #print(rev_key, "present")
                    df2 = dfs.get(rev_key)
                    df1rev = df1.copy().rename(columns={'object': 'subject', 'subject': 'object', 
                                        'object_category' : 'subject_category', 'subject_category' : 'object_category' })
                    df2rev = df2.copy().rename(columns={'object': 'subject', 'subject': 'object', 
                                        'object_category' : 'subject_category', 'subject_category' : 'object_category' })
        
                    # Add rows from df2rev to df1
                    new_rows = df2rev.merge(df1[['object','subject']], on = ['object','subject'], how='left', indicator=True)
                    new_rows = new_rows[new_rows['_merge'] == 'left_only'].drop(columns='_merge')
                    
                    df1 = pd.concat([df1, new_rows], ignore_index=True)
                    dfs[k] = df1
    
                    # Add rows from df1rev to df2 ?
                    new_rows = df1rev.merge(df2[['object','subject']], on = ['object','subject'], how='left', indicator=True)
                    new_rows = new_rows[new_rows['_merge'] == 'left_only'].drop(columns='_merge')
                    #print("newrows df1rev: ", new_rows.shape[0])
                    #print("fullreverse df1rev :",df1rev.shape[0])
                    df2 = pd.concat([df2, new_rows], ignore_index=True)
                    dfs[rev_key] = df2
                else:
                    #print(rev_key, " not present")
                    df1rev = df1.copy().rename(columns={'object': 'subject', 'subject': 'object', 
                                        'object_category' : 'subject_category', 'subject_category' : 'object_category' })
                    dfsnew[rev_key] = df1rev    
    
    dfs.update(dfsnew)
    
    # Add edges to heterodata structure. 
    # TODO: We will add the edge features some other day
    for k in dfs:
        current_object_category = k[0]
        current_subject_category = k[2]
        id_to_idx_for_current_object_category = id_to_category_idces[current_object_category]
        id_to_idx_for_current_subject_category = id_to_category_idces[current_subject_category]
        current_df = dfs[k]
        src_ids = current_df['object']
        tgt_ids = current_df['subject']
        src_ids_idx = [id_to_idx_for_current_object_category[e] for e in src_ids]
        tgt_ids_idx = [id_to_idx_for_current_subject_category[e] for e in tgt_ids]
    
        graph[k[0],k[1],k[2]].edge_index = torch.tensor([src_ids_idx, tgt_ids_idx],dtype=torch.long)



    #print(graph.metadata)
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


    
def get_biological_inverse_if_exists(predicate):
    inverse_edge_lookup = {'biolink:causes':'biolink:cuased_by',
                           'biolink:caused_by':'biolink:causes',
                           'biolink:interacts_with':'biolink:interacts_with',
                           'biolink:homologous_to':'biolink:homologous_to',
                           'biolink:orthologous_to':'biolink:orthologous_to',
                           'biolink:same_as':'biolink:same_as',
                           'biolink:related_to':'biolink:related_to',
                           'biolink:colocalizes_with':'biolink:colocalizes_with',
                           }
    inv = inverse_edge_lookup.get(predicate)
    if inv is None:
        inv = predicate + '_rev'

    return inv


#model = OntologyRGCN(
##    num_nodes=homo_data.num_nodes,
#    num_relations=len(data.edge_types)
#)


    
    #print(go_to_go_edge_gropus)
    #print(gene_to_go_edge_gropus)
    #graph["go",relation,"go"].edge_index = torch.tensor([go_to_idx(key), go_to_idx(target)],dtype=torch.long)
    #graph["gene",relation,"go"].edge_index = torch.tensor([gene_to_idx(key), go_to_idx(target)],dtype=torch.long)

    
    


