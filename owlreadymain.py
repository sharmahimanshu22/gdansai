from owlready2 import *
from owlready2 import Restriction, ThingClass
from collections import defaultdict, deque
import os
from elkreasoner import *
from relationnormalizer import *

rn = RelationNormalizer()

def get_ontology_from_filepath(ontology_file_path):
    onto = get_ontology(ontology_file_path).load()
    return onto

def get_nodes(onto):

    nodes = {}

    definition_prop = onto.search_one(iri="*IAO_0000115")
    exact_syn_prop = onto.search_one(iri="*hasExactSynonym")
    related_syn_prop = onto.search_one(iri="*hasRelatedSynonym")
    broad_syn_prop = onto.search_one(iri="*hasBroadSynonym")
    narrow_syn_prop = onto.search_one(iri="*hasNarrowSynonym")
    namespace_prop = onto.search_one(iri="*hasOBONamespace")
    subset_prop = onto.search_one(iri="*inSubset")
    comment_prop = onto.search_one(iri="*comment")
    xref_prop = onto.search_one(iri="*hasDbXref")

    for cls in onto.classes():

        node = defaultdict(list)

        node["id"] = cls.name

        node["label"] = (
            cls.label[0]
            if cls.label
            else ""
        )

        node["definition"] = (
            definition_prop[cls][0]
            if definition_prop[cls]
            else ""
        )

        node["exact_synonyms"] = list(
            exact_syn_prop[cls]
        )

        node["related_synonyms"] = list(
            related_syn_prop[cls]
        )

        node["broad_synonyms"] = list(
            broad_syn_prop[cls]
        )

        node["narrow_synonyms"] = list(
            narrow_syn_prop[cls]
        )

        node["namespace"] = (
            namespace_prop[cls][0]
            if namespace_prop[cls]
            else ""
        )

        node["comment"] = (
            comment_prop[cls][0]
            if comment_prop[cls]
            else ""
        )

        node["subsets"] = [
            str(x)
            for x in subset_prop[cls]
        ]

        node["xrefs"] = [
            str(x)
            for x in xref_prop[cls]
        ]

        
        for item in cls.is_a:
            if isinstance(item, ThingClass):
                if (item.name == "Thing"):
                    continue
                node['neighbors'].append((item.name, "is_a"))
            elif isinstance(item, Restriction):
                relation = item.property.label[0].replace(' ', '_')
                if hasattr(item.value, "name"):
                    node['neighbors'].append((item.value.name, relation))
                else:
                    continue

        
        assert(len(cls.equivalent_to) <= 1)
        for eq in cls.equivalent_to:   # eq is And
            for item in eq.Classes:    # conjuncts
                if isinstance(item, ThingClass):
                    node['equivalent'].append((item.name, "is_a"))
                elif isinstance(item, Restriction):
                    relation = item.property.label[0].replace(' ', '_')
                    if hasattr(item.value, "name"):
                        node['equivalent'].append((item.value.name, relation))
                    else:
                        continue
        

        nodes[cls.name] = node

    normalize_ontology(nodes, onto)

    return nodes



def normalize_ontology(nodes, ontology):
    invpropdict = defaultdict(lambda: None)
    invpropdict['is_a'] = 'inv_is_a'

    for prop in ontology.object_properties():
        proplabel = prop.label[0].replace(' ', '_')
        invprop = prop.inverse_property
        if invprop is not None:
            invpropdict[proplabel] = invprop.label[0].replace(' ', '_')
        else:
            invpropdict[proplabel] = 'inv_' + proplabel

    for k,v in nodes.items():
        neighbors = v['neighbors']
        for e in neighbors:
            tgt = e[0]
            relname = e[1]
            invrelname = invpropdict[relname]
            if invrelname is not None:
                tgtnode = nodes[tgt]
                tgtneighbors = tgtnode['neighbors']
                invrelation = (k, invrelname)
                #print(k, relname, tgt, "forward relation")
                #for e in tgtneighbors:
                    #print(tgt, e[1], e[0], " backward relation")
                if invrelation in tgtneighbors:
                    #print("invrelation exists")
                    continue
                else:
                    tgtneighbors.append(invrelation)
                    #print("invrelation added", invrelation)
                #print("\n")






def dist(startid, endid):
    q = deque()
    q.append((startid, 0))
    while(q):
        (currid, d) = q.popleft()
        print(currid)
        if (currid == 'Thing'):
            continue
        currnode = nodes[currid]
        neighbors = currnode['neighbors']
        print("neighbors: ", neighbors)
        for e in neighbors:
            if e[0] == endid:
                return d + 1
            else:
                q.append((e[0], d+1))
    return None


#PathDataset, ElkReasonerFactory, MOWLReasoner  = initialize_mowl()

#inferred_edges = get_elk_inferred_edges(os.path.abspath("go.owl"), PathDataset, ElkReasonerFactory, MOWLReasoner )

#for inf_edge in inferred_edges[:200]:
#    start = inf_edge[0]
#    end = inf_edge[1]
#    d = dist(start,end)
#    print(d)



## NODES DONE



'''
inverse_map = {
    "BFO_0000050": "BFO_0000051",
    "BFO_0000051": "BFO_0000050",
}


edges_augmented = []

for s, r, o in edges:

    edges_augmented.append((s, r, o))

    if r in inverse_map:
        edges_augmented.append(
            (o, inverse_map[r], s)
        )

'''





