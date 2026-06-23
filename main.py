from owlreadymain import *
from gnn import *
from loadgaf import *
import os
from models import HGTModel
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric
from train import *
torch_geometric.typing.WITH_PYG_LIB = False

import torch_geometric.transforms as T


# Assume we want to predict the directed edge: ("user", "buys", "item")

split_transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    edge_types= [('biolink:Disease', 'biolink:causes', 'biolink:Gene')],
    rev_edge_types = [( 'biolink:Gene', 'biolink:caused_by', 'biolink:Disease')],
    # Map the target type and its reverse so both are hidden during supervision

    disjoint_train_ratio=0.3,
    neg_sampling_ratio=0.0
)



# 2. Validation Loader (Static negatives from the split)
def get_eval_loader(val_data, target_edge_type):
    edge_label_index = val_data[target_edge_type].edge_label_index
    
    # 2. Pull the labels (RandomLinkSplit automatically creates 1s for positive, 0s for negative)
    edge_label = val_data[target_edge_type].edge_label
    
    val_loader = LinkNeighborLoader(
        data=val_data,
        edge_label_index=(target_edge_type, edge_label_index),
        edge_label=edge_label,
        neg_sampling_ratio=1.0,  # Keep this to sample additional negatives per batch
        num_neighbors={etype: [3] for etype in val_data.edge_types},
        batch_size=128,
        shuffle=True
    )

   
    return val_loader


def get_train_loader(data, target_edge_type):

    edge_label_index = data[target_edge_type].edge_label_index
    
    # 2. Pull the labels (RandomLinkSplit automatically creates 1s for positive, 0s for negative)
    edge_label = data[target_edge_type].edge_label
    
    loader = LinkNeighborLoader(
        data=data,
        edge_label_index=(target_edge_type, edge_label_index),
        edge_label=edge_label,
        neg_sampling_ratio=1.0,  # Keep this to sample additional negatives per batch
        num_neighbors={etype: [3] for etype in data.edge_types},
        batch_size=128,
        shuffle=False
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
    tgt_edge_type = ('biolink:Disease', 'biolink:causes', 'biolink:Gene')

    graph = torch.load('hetero_data.pt', weights_only=False)
    print(graph.metadata)
    train_data, val_data, test_data = split_transform(graph)

    etype = ('biolink:Disease', 'biolink:causes', 'biolink:Gene')

    #print(train_data[etype])
    #print(val_data[etype])

    #print(train_data[etype].keys())
    #print(val_data[etype].keys())

    #for etype in train_data.edge_types:
    #    if not hasattr(train_data[etype], 'edge_index'):
    #        print("MISSING:", etype)


    model = HGTModel(graph)
    model = model.to(device)
    predictor = LinkPredictor(128)
    predictor = predictor.to(device)
    train_loader = get_train_loader(train_data,tgt_edge_type)
    eval_loader = get_eval_loader(val_data, tgt_edge_type)
    test_loader = get_eval_loader(test_data, tgt_edge_type)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=1e-3)


    num_epochs = 20
    best_val_auc = 0.0

    for epoch in range(1, num_epochs + 1):
        # Pass 'device' tracking directly into execution functions
        loss = train(model,train_loader, predictor, optimizer, tgt_edge_type, device)
        val_auc = evaluate(model,eval_loader, predictor, tgt_edge_type, device)
        
        print(f"Epoch {epoch:02d} | Train Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model': model.state_dict(),
                'predictor': predictor.state_dict()
            }, 'best_model.pt')

    # --- Final Testing ---
    checkpoint = torch.load('best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model'])
    predictor.load_state_dict(checkpoint['predictor'])

    test_auc = evaluate(test_loader, target_edge_type, device)
    print(f"Final Test Performance -> Test AUC: {test_auc:.4f}")



def main2():
    
    device = torch.device('cuda')

    graph = create_hetero_graph_from_monarch()
    torch.save(graph, "hetero_data.pt")
    return

    #graph = graph.to(device)

    

    #return
    # To load the object back into memory
    tgt_edge_type = ('biolink:Disease', 'biolink:causes', 'biolink:Gene')

    graph = torch.load('hetero_data.pt', weights_only=False)
    train_data, val_data, test_data = split_transform(graph)

    model = HGTModel(graph)
    model = model.to(device)
    train_loader = get_train_loader(train_data,tgt_edge_type)
    eval_loader = get_eval_loader(val_data, tgt_edge_type)
    predictor = LinkPredictor(128)
    predictor = predictor.to(device)
    model.train()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=1e-3)
    
    for epoch in range(100):
        totalloss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            z_dict = model(batch)
            src, dst = batch[tgt_edge_type]['edge_label_index']

            gene_emb = z_dict['biolink:Gene'][dst]
            disease_emb = z_dict['biolink:Disease'][src]

            pred = predictor(gene_emb, disease_emb)

            edge = (pred > 0).int()

            loss = F.binary_cross_entropy_with_logits(pred,batch[tgt_edge_type]['edge_label'].float())
            totalloss = totalloss + loss
            loss.backward()
            optimizer.step()
            #print(edge, type(edge))
            #print(batch[etype]['edge_label'], type(batch[etype]['edge_label']))
            correct = (edge == batch[tgt_edge_type]['edge_label']).float()

            accuracy = correct.mean()
            print("Accuracy: ", accuracy)

        print("Loss : ", totalloss.item())
        

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