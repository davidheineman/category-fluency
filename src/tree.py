import networkx as nx
from .similarity import glove_pairwise_similarity


def create_animal_graph(exemplars, glove_model, word_freq, epsilon=0.4):
    exemplars = [r for r in exemplars if r.lower() in glove_model.keys()]
    dist = glove_pairwise_similarity(exemplars, glove_model)

    G = create_graph(exemplars, dist)

    # Prune edges with weight less than e
    G = prune_edges(G, epsilon)

    # Add 'animal' root node using word frequency
    G = add_animal_node(G, word_freq)

    return G


def create_norm_graph(exemplars, norms):
    exemplars = [r for r in exemplars if r.lower() in norms.keys()]

    # Create an edge list, where an edge only exists if there it occurs in the norm list
    # for that node
    edges = [[0 for _ in range(len(exemplars))] for _ in range(len(exemplars))]
    for i1, e1 in enumerate(exemplars):
        for i2, e2 in enumerate(exemplars):
            if e1 not in norms.keys() or e2 not in norms[e1]:
                edges[i1][i2] = 0
            else:
                edges[i1][i2] = 1

    G = create_graph(exemplars, edges)

    # Prune edges that shouldn't exist
    G = prune_edges(G, 1e-4)

    return G


def create_graph(nodes, weights):
    """
    Create a networkx graph given a set of nodes and edge weights
    """
    edges = []
    for i in range(len(weights)):
        for j in range(len(weights)):
            if i != j:
                edges += [
                    (nodes[i], nodes[j], weights[i][j])
                ]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G


def prune_edges(graph, epsilon):
    """
    Delete all edges in the graph where G(u, v) < epsilon
    """
    edges_to_remove = [(u, v) for u, v, d in graph.edges(data=True) if 'weight' in d and d['weight'] < epsilon]
    graph.remove_edges_from(edges_to_remove)
    return graph


def add_animal_node(graph, word_frequency):
    """
    Add a fully connected 'animal' node whose weights are given by word frequency
    """
    graph.add_node('animal')
    total_word_freq = sum(word_frequency.values())
    for node in graph.nodes():
        if node != 'animal':
            graph.add_edge('animal', node, weight=word_frequency[node]/total_word_freq)
    return graph


def mst(typicality_sequence, bert_distances, root_node=None):
    """
    Convert pairwise distances to a MST
    """
    G = create_graph(typicality_sequence, bert_distances)
    MST = nx.minimum_spanning_tree(G)
    if root_node is None:
        root_node = typicality_sequence[0]
    pos = {typicality_sequence[i]: (i, -nx.shortest_path_length(MST, source=root_node, target=typicality_sequence[i])) for i in range(len(typicality_sequence))}
    return MST, pos
