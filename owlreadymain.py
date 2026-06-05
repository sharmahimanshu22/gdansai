from owlready2 import *
from owlready2 import Restriction, ThingClass
from collections import defaultdict, deque
import os
from elkreasoner import *

onto = get_ontology("go.owl").load()


edges = []

eq_edges = []

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
            node['neigbors'].append((item.name, "is_a"))
        elif isinstance(item, Restriction):
            relation = item.property.name
            if hasattr(item.value, "name"):
                node['neigbors'].append((item.value.name, relation))
            else:
                continue

    
    assert(len(cls.equivalent_to) <= 1)
    for eq in cls.equivalent_to:   # eq is And
        for item in eq.Classes:    # conjuncts
            if isinstance(item, ThingClass):
                node['equivalent'].append((item.name, "is_a"))
            elif isinstance(item, Restriction):
                relation = item.property.name
                if hasattr(item.value, "name"):
                    node['equivalent'].append((item.value.name, relation))
                else:
                    continue
    



    nodes[cls.name] = node



#for k,v in nodes.items():
#    print(k,": ",v)


 

def dist(startid, endid):
    q = deque()
    q.append((startid, d))
    while(q):
        currid, d = q.popleft()
        currnode = nodes[currid]
        neighbors = currnode['neighbors']
        for e in neighbors:
            if e[0] == endid:
                return d + 1
            else:
                q.append((e, d+1))
    return None


print("done")

PathDataset, ElkReasonerFactory, MOWLReasoner  = initialize_mowl()

inferred_edges = get_elk_inferred_edges(os.path.abspath("go.owl"), PathDataset, ElkReasonerFactory, MOWLReasoner )





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





