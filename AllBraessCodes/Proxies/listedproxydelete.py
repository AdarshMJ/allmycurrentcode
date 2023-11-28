import networkx as nx
import scipy.sparse as sp
import numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
eps = 1e-6

def proxy_delete(i,j,deg, vecsold, gap):
    deg = deg[:, np.newaxis]
    vecsold = np.divide(vecsold, np.sqrt(deg))
    vecsold = vecsold / np.linalg.norm(vecsold, axis=1)[:, np.newaxis]
    delta_gap = -((vecsold[i,1]-vecsold[j,1])**2-gap*(vecsold[i,1]**2 + vecsold[j,1]**2))
    index_best_edge = np.argmax(delta_gap)
    return [i][index_best_edge],[j][index_best_edge],delta_gap

def return_existing_edges(G):
  return (list(G.edges))

def obtain_Lnorm(G):
    adj = nx.adjacency_matrix(G)
    adj = adj.tolil()
    adj.setdiag(1)
    adj= adj.tocsr()
    deg = np.array(adj.sum(axis=1)).flatten()
    D_sqrt_inv = sp.diags(np.power(deg, -0.5))
    L_norm = (sp.eye(adj.shape[0]) - D_sqrt_inv @ adj @ D_sqrt_inv)
    return deg,L_norm

def spectral_gap(L_norm):
        gap, vectors = sp.linalg.eigsh(L_norm, k=2,sigma=0.1,which='LM')
        return gap[1], vectors

def update_gap(vecs,L_norm):
        gap, vectors = sp.linalg.eigsh(L_norm, k=2,sigma=0,which='LM',v0 = vecs[:,1])
        return vectors

def update_Lnorm(u,v,L_norm,deg):
      L_norm[u, :] *= np.sqrt(deg[u] / (deg[u] - 1))
      L_norm[:, u] = L_norm[u, :].T
      L_norm[v, :] *= np.sqrt(deg[v] / (deg[v] - 1))
      L_norm[:, v] = L_norm[v, :].T
      deg[u] -= 1
      deg[v] -= 1
      L_norm[u,u] = 1-1/deg[u]
      L_norm[v,v] = 1-1/deg[v]
      L_norm[u, v] = 0
      L_norm[v, u] = 0

      return deg,L_norm

def process_edges(g):
    existing_edges = return_existing_edges(g)
    deg, L_norm = obtain_Lnorm(g)
    gap, vecs = spectral_gap(L_norm)
    edge_dgap_mapping = {}
    for edges in tqdm(existing_edges):
        i, j = edges
        u, v, dgap = proxy_delete(i, j, deg, vecs, gap)
        v, u = u, v
        deg, L_norm = update_Lnorm(u, v, L_norm, deg)
        _, vecs = spectral_gap(L_norm)
        edge_dgap_mapping[(u, v)] = dgap

    sorted_edges = sorted(edge_dgap_mapping.items(), key=lambda x: x[1], reverse=True)

    return sorted_edges


def process_and_update_edges(g, k):
    print("Processing Edges...")
    best_edges = process_edges(g)
    print("Done!")
    print(f"Deleting {k} edges...")

    deleted_edges_count = 0  # Counter for deleted edges

    for i in range(len(best_edges)):
        edges, dgap = best_edges[i]
        s, t = edges
        print("Deleting edge", (s, t))
        # Try removing the edge
        g.remove_edge(s, t)

        # Check if the graph is disconnected
        if not nx.is_connected(g):
            print("Graph disconnected. Putting edge back", (s, t))
            g.add_edge(s, t)
        else:
            # Increment the deleted edges count
            deleted_edges_count += 1

        if deleted_edges_count == int(k/2):
              print("Updating the criterion")
              best_edges = process_edges(g)
              for edge, updated_dgap in best_edges:
                      if edge == (s, t):
                          dgap = updated_dgap
                          break
        # Check if we have deleted the required number of edges
        if deleted_edges_count >= k:
            break
    print(f"Edges deleted = {deleted_edges_count}")
    return g