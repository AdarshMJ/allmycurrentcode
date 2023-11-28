import random
import numpy as np
random.seed(44)
np.random.seed(44)
import networkx as nx

def spectral_gap(G):
    Lmod = nx.normalized_laplacian_matrix(G).todense()
    valsmod, vecsmod = np.linalg.eigh(Lmod)
    return valsmod[1]

def greedy_add(G):
    initial_gap = spectral_gap(G)
    print("Initial spectral gap:", initial_gap)
    gaps = [initial_gap]

    while True:
        max_gap_increase = 0
        best_edge_to_add = None

        for node1 in G.nodes():
            for node2 in G.nodes():
                if node1 != node2 and not G.has_edge(node1, node2):
                    temp_G = G.copy()
                    temp_G.add_edge(node1, node2)
                    new_gap = spectral_gap(temp_G)
                    gap_increase = new_gap - initial_gap

                    if gap_increase > max_gap_increase:
                        max_gap_increase = gap_increase
                        best_edge_to_add = (node1, node2)

        if best_edge_to_add:
            G.add_edge(*best_edge_to_add)
            initial_gap = spectral_gap(G)
            gaps.append(initial_gap)
            print(f"Added edge {best_edge_to_add}, current spectral gap: {initial_gap}")
        else:
            break

    return gaps