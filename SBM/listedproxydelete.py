import networkx as nx
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
eps = 1e-6

def npgap(G):
  Lmod = nx.normalized_laplacian_matrix(G).todense()
  valsmod,vecsmod = np.linalg.eigh(Lmod)
  
  return (round(valsmod[1], 4))

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
    D_sqrt_inv = sp.diags(np.power(deg+eps, -0.5))
    L_norm = (sp.eye(adj.shape[0]) - D_sqrt_inv @ adj @ D_sqrt_inv)
    return deg,L_norm

def spectral_gap(L_norm):
        gap, vectors = sp.linalg.eigsh(L_norm, k=2,sigma = 0.1,which='LM',ncv=10000)
        return gap[1], vectors

# def update_gap(vecs,L_norm):
#         gap, vectors = sp.linalg.eigsh(L_norm, k=2,sigma=0,which='LM',v0 = vecs[:,1])
#         return vectors

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


def process_and_update_edges(g):
    best_edges = process_edges(g)
    gap_holder = []
    previous_gap = npgap(g)

    for edges, dgap in best_edges:
        s, t = edges
        print("Deleting edge", (s, t))
        print("Dgap value", dgap)
        g.remove_edge(s, t)

        #Check if the graph is disconnected
        if not nx.is_connected(g):
           print("Graph disconnected. Putting edge back", (s, t))
           g.add_edge(s, t)
           continue

        newgap = npgap(g)
        print("Spectral gap after deletion:", newgap)
        print("=" * 40)

        if newgap > previous_gap:
            gap_holder.append(newgap)
            previous_gap = newgap

            # Recalculate best edges and update dgap
            print("updating the best edges...")
            best_edges = process_edges(g)
            for edge, updated_dgap in best_edges:
                if edge == (s, t):
                    dgap = updated_dgap
                    break

        else:
            g.add_edge(s, t)
            print("Putting edge back", (s, t))
            print("Spectral gap after adding edge back:", previous_gap)
            print("=" * 40)

    # Print the final gap_holder
    print("Final gap_holder:", gap_holder)
    return gap_holder,g

# def process_and_update_edges(g, k):
#     gap_holder = []
#     print("Processing Edges...")
#     best_edges = process_edges(g)
#     print("Done!")
#     print(f"Deleting {k} edges...")
#     previous_gap = npgap(g)

#     deleted_edges_count = 0  # Counter for deleted edges
#     update_criterion_after = 100  # Number of edges to delete before updating the criterion
#     criterion_update_counter = 0  # Counter for updating the criterion

#     for edges, dgap in best_edges:
#         s, t = edges
#         print("Deleting edge", (s, t))
#         g.remove_edge(s, t)

#         if not nx.is_connected(g):
#             print("Graph disconnected. Putting edge back", (s, t))
#             g.add_edge(s, t)
#         else:
#             deleted_edges_count += 1
#             new_gap = npgap(g)
#             print(f"Spectral Gap after deleting {s,t} {new_gap}")

#             if new_gap > previous_gap:
#                 gap_holder.append(new_gap)
#                 previous_gap = new_gap
#                 print("============" * 40)
#             else:
#                 print(f"Gap Decreased. Putting edge {s,t} back")
#                 g.add_edge(s, t)
#                 print(f"Gap restored to {previous_gap}")
#                 deleted_edges_count -= 1
#                 print("============" * 40)

#         criterion_update_counter += 1

#         if criterion_update_counter == update_criterion_after:
#             print("Updating the criterion")
#             best_edges = process_edges(g)
#             for edge, updated_dgap in best_edges:
#                       if edge == (s, t):
#                           dgap = updated_dgap
#                           break
#             criterion_update_counter = 0  # Reset the update counter

#         print("============" * 40)
#         print()

#         if deleted_edges_count >= k:
#             break
#     print(f"Edges deleted = {deleted_edges_count}")
#     return g, gap_holder
