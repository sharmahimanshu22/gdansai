print("before import mowl")
import mowl



def initialize_mowl():
    print("before init")
    mowl.init_jvm("4g")

    print("after init")

    print("before PathDataset")
    from mowl.datasets.base import PathDataset

    print("before ELK")
    from org.semanticweb.elk.owlapi import ElkReasonerFactory

    print("before reasoner")
    from mowl.reasoning import MOWLReasoner

    print("done")

    return PathDataset, ElkReasonerFactory, MOWLReasoner 

def get_go_id(cls):
    iri = cls.getIRI().toString()
    return iri.split("/")[-1]



def get_elk_inferred_edges(ontology_file_path,PathDataset, ElkReasonerFactory, MOWLReasoner):


    ds = PathDataset(ontology_file_path)
    onto = ds.ontology

    factory = ElkReasonerFactory()
    reasoner = factory.createReasoner(onto)

    mowl_reasoner = MOWLReasoner(reasoner)

    classes = list(onto.getClassesInSignature())

    subclass_axioms = mowl_reasoner.infer_subclass_axioms(classes)

    inferred_edges = []
    for ax in subclass_axioms:
        sub = get_go_id(ax.getSubClass())
        sup = get_go_id(ax.getSuperClass())
        inferred_edges.append((sub, sup))
    return inferred_edges


