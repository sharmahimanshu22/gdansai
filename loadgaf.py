

def get_gene_nodes():
    gene_nodes = {}


    with open("goa_human.gaf") as f:

        for line in f:

            if line.startswith("!"):
                continue

            cols = line.rstrip("\n").split("\t")

            gene = cols[1]       # DB Object ID
            relation = cols[3]   # Relation
            go_id = cols[4].replace(":", "_")

            gene_node = {"id": gene, "symbol": cols[2], "name": cols[9], "type": cols[11], "taxon": cols[12], "neighbors": [(go_id, relation)]}

            gene_nodes[gene] = gene_node

    return gene_nodes