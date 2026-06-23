from owlreadymain import *
from gnn import *
from loadgaf import *
import os
from models import HGTModel
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric

torch_geometric.typing.WITH_PYG_LIB = False


def getLinkLoader(data, target_edge_type):

    edge_index = data[target_edge_type].edge_index

    loader = LinkNeighborLoader(
        data=data,
        # ✅ YOUR VERSION EXPECTS THIS FORMAT
        edge_label_index=(target_edge_type, edge_index),
        edge_label=torch.ones(edge_index.size(1)),
        neg_sampling_ratio=1.0,
        num_neighbors={etype: [3] for etype in data.edge_types},
        batch_size=128,
        shuffle=True
    )

    return loader


class LinkPredictor(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_src, x_dst):

        x = torch.cat([x_src, x_dst], dim=-1)

        return self.lin(x).squeeze(-1)
    


def main():
    
    device = torch.device('cuda')

    #graph = create_hetero_graph_from_monarch()
    #graph = graph.to(device)

    #torch.save(graph, "hetero_data.pt")

    #return
    # To load the object back into memory
    graph = torch.load('hetero_data.pt', weights_only=False)
    model = HGTModel(graph)
    model = model.to(device)        
    loader = getLinkLoader(graph, ('biolink:Disease', 'biolink:causes', 'biolink:Gene'))
    predictor = LinkPredictor(128)
    predictor = predictor.to(device)

    for batch in loader:

        batch = batch.to(device)
        z_dict = model(batch)
        etype = ('biolink:Disease', 'biolink:causes', 'biolink:Gene')
        src, dst = batch[etype]['edge_label_index']

        print("src: ", src )
        print("dst: ", dst )

        print("SHAPE")
        print(z_dict['biolink:Gene'].shape)
        print(z_dict['biolink:Disease'].shape)

        print(src.min(), src.max())
        print(dst.min(), dst.max())

        print("NUM NODES: ")
        print(batch['biolink:Gene'].num_nodes)
        print(batch['biolink:Disease'].num_nodes)

        print(z_dict['biolink:Gene'][dst])
        print(z_dict['biolink:Disease'][src])

        gene_emb = z_dict['biolink:Gene'][dst]
        disease_emb = z_dict['biolink:Disease'][src]

        print("embeddings\n")
        print(gene_emb)
        print(disease_emb)

        pred = predictor(gene_emb, disease_emb)

        print("PRED SHAPE")
        print(pred.shape)

        print(batch[etype]['edge_label'].shape)

        loss = F.binary_cross_entropy_with_logits(
            pred,
            batch[etype]['edge_label'].float()
        )
        print(loss)
        loss.backward()
    return


    onto = get_ontology_from_filepath(os.path.abspath("go.owl"))
    go_nodes = get_nodes(onto)
    gene_nodes = get_gene_nodes()
    graph = create_hetero_graph(go_nodes, gene_nodes)

    for edge_type in graph.edge_types:
        edge_index = graph[edge_type].edge_index

        print(edge_type)
        print(type(edge_index))
        print(edge_index.shape)
        print(edge_index.dtype)
        print("\n")



    #homo_data = graph.to_homogeneous()

    #print(graph.edge_types)
    #print(graph.node_types)

    #num_nodes = homo_data.num_nodes

    #embedding = torch.nn.Embedding(
    #    num_nodes,
    #    256
    #)

    print(graph.metadata)
    print("edge_index_dict: ", graph.edge_index_dict)

    sys.exit(0)
    model = OntologyModel(graph)

    out = model(graph)

    for k, v in out.items():
        print(k, v.shape)

    print(out)




if __name__ == "__main__":
    main()





#model = OntologyRGCN(
#    num_nodes=homo_data.num_nodes,
#    num_relations=len(graph.edge_types)
#    )

#embeddings = model(
#    homo_data.edge_index,
#    homo_data.edge_type
#    )