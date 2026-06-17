import pandas as pd
from itertools import islice


KG_NODES_FILE = '/hpc/users/sharmh15/labhome/projects/genediseaseassociation/monarch/monarchkg/monarch-kg_nodes.tsv'
KG_EDGES_FILE = '/hpc/users/sharmh15/labhome/projects/genediseaseassociation/monarch/monarchkg/monarch-kg_edges.tsv'
RELEVANT_NODE_FIELDS = ['id', 'category','name', 'type', 'description' , 'full_name' , 'in_taxon',  'in_taxon_label',  'symbol',  'synonym', 
     'exact_synonym', 'broad_synonym', 'narrow_synonym',   'related_synonym',  'deprecated' , 'subsets',  'has_gene', 'has_biological_sex']
#has_attribute is always None
#same_as is always None
#synonyms is always None

RELEVANT_EDGE_FIELDS = ['id', 'predicate', 'category', 'agent_type', 'knowledge_level',
                        'negated',
                        'has_evidence', 
                        'onset_qualifier',  'frequency_qualifier',  'sex_qualifier', 'stage_qualifier', 'species_context_qualifier' ,
                        'disease_context_qualifier',
                        'has_count', 'has_percentage', 'has_quotient', 'has_total',
                        'subject', 'object'
                        ]

IRRELEVANT_EDGE_FIELDS = ['aggregator_knowledge_source',  'original_predicate', 'primary_knowledge_source', 'file_source', 'provided_by', 'object_category',
                        'subject_category', 'publications', 'object_aspect_qualifier', 'knowledge_source', 'original_subject', 'original_object',
                        'qualifier', 'qualifiers', 'object_specialization_qualifier', 'has_attribute', 'FDA_adverse_event_level']


def load_monarch_to_dictionary_using_node_file():
    df = pd.read_csv(KG_NODES_FILE,sep='\t')
    print(df[['id', 'name', 'category', 'type']].head(20))
    return
    df = df[['id','category']]
    df['id'] = df['id'].str.split(":").str[0]
    df = df.drop_duplicates()
    pd.set_option('display.max_rows', 100)
    print(df.head(100))
    return
    df = df[RELEVANT_NODE_FIELDS]
    result = df.set_index('id').to_dict(orient='index')
    


def load_monarch_to_dictionary_using_edge_file():
    df = pd.read_csv(KG_EDGES_FILE,sep='\t')
    df = df[RELEVANT_EDGE_FIELDS]
    result = df.set_index('id').to_dict(orient='index')
    
    few_elements = dict(islice(result.items(), 3))
    print(few_elements)






load_monarch_to_dictionary_using_node_file()








#for col in df.columns:
#        df[[col]].to_csv(f"/hpc/users/sharmh15/labhome/projects/genediseaseassociation/monarch/monarchkg/edge_{col}.csv", index=False)