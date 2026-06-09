from owlreadymain import *
from gnn import *
from loadgaf import *
import os


def main():
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
    model = OntologyModel(graph)
    #model = OntologyRGCN(
    #    num_nodes=homo_data.num_nodes,
    #    num_relations=len(graph.edge_types)
    #    )
    
    #embeddings = model(
    #    homo_data.edge_index,
    #    homo_data.edge_type
    #    )


    out = model(graph)

    for k, v in out.items():
        print(k, v.shape)

    print(embeddings)




if __name__ == "__main__":
    main()