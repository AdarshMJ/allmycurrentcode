import random
import numpy as np
random.seed(44)
np.random.seed(44)
import networkx as nx

def spectral_gap(G):
    Lmod = nx.normalized_laplacian_matrix(G).todense()
    valsmod, vecsmod = np.linalg.eigh(Lmod)
    return valsmod[1]

def greedydelete(G):
    initial_gap = spectral_gap(G)
    print("Initial spectral gap:", initial_gap)
    gaps = [initial_gap]

    while G.edges():
        max_gap_increase = 0
        best_edge_to_remove = None

        for edge in G.edges():
            temp_G = G.copy()
            temp_G.remove_edge(*edge)
            new_gap = spectral_gap(temp_G)
            gap_increase = new_gap - initial_gap

            if gap_increase > max_gap_increase:
                max_gap_increase = gap_increase
                best_edge_to_remove = edge

        if best_edge_to_remove:
            G.remove_edge(*best_edge_to_remove)
            initial_gap = spectral_gap(G)
            gaps.append(initial_gap)
            print(f"Removed edge {best_edge_to_remove}, current spectral gap: {initial_gap}")
        else:
            break

    return gaps