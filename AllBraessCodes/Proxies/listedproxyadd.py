# import networkx as nx
# import scipy.sparse as sp
# import numpy as np
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings('ignore')
# eps = 1e-6


# def npgap(G):
#   Lmod = nx.normalized_laplacian_matrix(G).todense()
#   valsmod,vecsmod = np.linalg.eigh(Lmod)
#   return valsmod[1]


# # def proxy_add(missing_edges, deg,vecsold, gap):
# #     deg = deg[:, np.newaxis] 
# #     vecsold = np.divide(vecsold, np.sqrt(deg))
# #     vecsold = vecsold / np.linalg.norm(vecsold, axis=1)[:, np.newaxis]
# #     delta_gap = (vecsold[missing_edges[0],1]-vecsold[missing_edges[1],1])**2-gap*(vecsold[missing_edges[0],1]**2 + vecsold[missing_edges[1],1]**2)
# #     index_best_edge = np.argmax(delta_gap)
# #     return missing_edges[0][index_best_edge],missing_edges[1][index_best_edge],index_best_edge,delta_gap

# def proxy_add(i,j, deg,vecsold, gap):
#     deg = deg[:, np.newaxis] 
#     vecsold = np.divide(vecsold, np.sqrt(deg))
#     vecsold = vecsold / np.linalg.norm(vecsold, axis=1)[:, np.newaxis]
#     delta_gap = (vecsold[i,1]-vecsold[j,1])**2-gap*(vecsold[i,1]**2 + vecsold[j,1]**2)
#     index_best_edge = np.argmax(delta_gap)
#     return [i][index_best_edge],[j][index_best_edge],index_best_edge, delta_gap

# ### for larger graphs use this ####
# # def return_missing_edges(G):
# #     adj = nx.adjacency_matrix(G)
# #     edges = (adj == 0).multiply(adj.T == 0)
# #     missing_edges = sp.triu(edges,k=1)
# #     candidates = missing_edges.nonzero() 
# #     i, j = candidates[0], candidates[1]
# #     return i,j 

# #### for smaller graphs ####
# def return_missing_edges(G):
#   gc = nx.complement(G)
#   missing_edges = list(gc.edges) 
#   return missing_edges

# def obtain_Lnorm(G):
#     adj = nx.adjacency_matrix(G)
#     adj = adj.tolil()
#     adj.setdiag(1)
#     adj= adj.tocsr()
#     deg = np.array(adj.sum(axis=1)).flatten()   
#     D_sqrt_inv = sp.diags(np.power(deg, -0.5))
#     L_norm = sp.eye(adj.shape[0]) - D_sqrt_inv @ adj @ D_sqrt_inv 
#     return deg,L_norm

# # def spectral_gap(L_norm):
# #     gap, vectors = sp.linalg.eigsh(L_norm, k=2, sigma=0, which='LM')
# #     return gap[1],vectors

# def spectral_gap(L_norm):
#         gap, vectors = sp.linalg.eigsh(L_norm, k=2, sigma=0.1,which='LM')
#         return gap[1], vectors


# # def update_gap(L_norm,vecsold):
# #     _, vecsnew = sp.linalg.eigsh(L_norm, k=2, sigma=0, which='LM',v0 = vecsold[:,1])
# #     return vecsnew

# def update_Lnorm(u,v,L_norm,deg):
#       L_norm[u, :] *= np.sqrt(deg[u] / (deg[u] + 1))
#       L_norm[:, u] = L_norm[u, :].T
#       L_norm[v,:] *= np.sqrt(deg[v]/(deg[v] + 1))
#       L_norm[:,v] = L_norm[v,:].T
#       L_norm[u,v] = -1/np.sqrt((deg[u]+1)*(deg[v]+1))
#       L_norm[v,u] = L_norm[u,v] 
#       deg[u]+=1
#       deg[v]+=1
#       L_norm[u,u] = 1-1/deg[u]
#       L_norm[v,v] = 1-1/deg[v]
#       return deg,L_norm

# def add_new_edges(g):
#     missing_edges = return_missing_edges(g)
#     deg, L_norm = obtain_Lnorm(g)
#     gap, vecs = spectral_gap(L_norm)
#     edge_dgap_mapping = {}
#     for edges in tqdm(missing_edges):
#             i, j = edges
#             u, v, index_best_edge,dgap = proxy_add(i,j,deg,vecs,gap)
#             v,u = u,v
#             deg,L_norm = update_Lnorm(u,v,L_norm,deg)
#             _, vecs = spectral_gap(L_norm)
#             i = np.delete(i, index_best_edge)
#             j = np.delete(j, index_best_edge) 
#             edge_dgap_mapping[(u, v)] = dgap

#     sorted_edges = sorted(edge_dgap_mapping.items(), key=lambda x: x[1], reverse=True)

#     return sorted_edges


# def add_and_update_edges(g):
#     best_edges = add_new_edges(g)
#     gap_holder = []
#     previous_gap = npgap(g)

#     for edges, dgap in best_edges:
#         s, t = edges
#         print("Adding edge", (s, t))
#         print("Dgap value", dgap)
#         g.add_edge(s, t)

#         newgap = npgap(g)
#         print("Spectral gap after addition :", newgap)
#         print("=" * 40)

#         if newgap > previous_gap:
#             gap_holder.append(newgap)
#             previous_gap = newgap

#             # Recalculate best edges and update dgap
#             print("updating the best edges...")
#             best_edges = add_new_edges(g)
#             for edge, updated_dgap in best_edges:
#                 if edge == (s, t):
#                     dgap = updated_dgap
#                     break

#         else:
#             g.remove_edge(s, t)
#             print("Deleting that edge ", (s, t))
#             print("Spectral gap after refraining from adding that edge:", previous_gap)
#             print("=" * 40)
#     # Print the final gap_holder
#     print("Final gap_holder:", gap_holder)
#     return gap_holder

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
  return valsmod[1]


# def proxy_add(missing_edges, deg,vecsold, gap):
#     deg = deg[:, np.newaxis] 
#     vecsold = np.divide(vecsold, np.sqrt(deg))
#     vecsold = vecsold / np.linalg.norm(vecsold, axis=1)[:, np.newaxis]
#     delta_gap = (vecsold[missing_edges[0],1]-vecsold[missing_edges[1],1])**2-gap*(vecsold[missing_edges[0],1]**2 + vecsold[missing_edges[1],1]**2)
#     index_best_edge = np.argmax(delta_gap)
#     return missing_edges[0][index_best_edge],missing_edges[1][index_best_edge],index_best_edge,delta_gap

def proxy_add(i,j, deg,vecsold, gap):
    deg = deg[:, np.newaxis] 
    vecsold = np.divide(vecsold, np.sqrt(deg))
    vecsold = vecsold / np.linalg.norm(vecsold, axis=1)[:, np.newaxis]
    delta_gap = (vecsold[i,1]-vecsold[j,1])**2-gap*(vecsold[i,1]**2 + vecsold[j,1]**2)
    index_best_edge = np.argmax(delta_gap)
    return [i][index_best_edge],[j][index_best_edge],index_best_edge, delta_gap

### for larger graphs use this ####
def return_missing_edges(G):
    adj = nx.adjacency_matrix(G)
    edges = (adj == 0).multiply(adj.T == 0)
    missing_edges = sp.triu(edges,k=1)
    candidates = missing_edges.nonzero() 
    i, j = candidates[0], candidates[1]
    return i,j 

#### for smaller graphs ####
# def return_missing_edges(G):
#   gc = nx.complement(G)
#   missing_edges = list(gc.edges) 
#   return missing_edges

def obtain_Lnorm(G):
    adj = nx.adjacency_matrix(G)
    adj = adj.tolil()
    adj.setdiag(1)
    adj= adj.tocsr()
    deg = np.array(adj.sum(axis=1)).flatten()   
    D_sqrt_inv = sp.diags(np.power(deg, -0.5))
    L_norm = sp.eye(adj.shape[0]) - D_sqrt_inv @ adj @ D_sqrt_inv 
    return deg,L_norm

def spectral_gap(L_norm):
        gap, vectors = sp.linalg.eigsh(L_norm, k=2, sigma=0.1,which='LM')
        return gap[1], vectors

def update_Lnorm(u,v,L_norm,deg):
      L_norm[u, :] *= np.sqrt(deg[u] / (deg[u] + 1))
      L_norm[:, u] = L_norm[u, :].T
      L_norm[v,:] *= np.sqrt(deg[v]/(deg[v] + 1))
      L_norm[:,v] = L_norm[v,:].T
      L_norm[u,v] = -1/np.sqrt((deg[u]+1)*(deg[v]+1))
      L_norm[v,u] = L_norm[u,v] 
      deg[u]+=1
      deg[v]+=1
      L_norm[u,u] = 1-1/deg[u]
      L_norm[v,v] = 1-1/deg[v]
      return deg,L_norm

def add_new_edges(g,k):
    missing_edges = return_missing_edges(g)
    deg, L_norm = obtain_Lnorm(g)
    gap, vecs = spectral_gap(L_norm)
    edge_dgap_mapping = {}
    for edges in tqdm(missing_edges[1:k+1]):
            i, j = edges
            u, v, index_best_edge,dgap = proxy_add(i,j,deg,vecs,gap)
            v,u = u,v
            deg,L_norm = update_Lnorm(u,v,L_norm,deg)
            _, vecs = spectral_gap(L_norm)
            i = np.delete(i, index_best_edge)
            j = np.delete(j, index_best_edge) 
            edge_dgap_mapping[(u, v)] = dgap

    sorted_edges = sorted(edge_dgap_mapping.items(), key=lambda x: x[1], reverse=True)

    return sorted_edges


def add_and_update_edges(g,k):
    print(f"Finding the best edges to add...")
    best_edges = add_new_edges(g)
    print("Done!")
  
    added_edges_count = 0  # Counter for deleted edges

    for edges, dgap in best_edges:
        s, t = edges
        print("Adding edge", (s, t))
        print("Dgap value", dgap)
        g.add_edge(s, t)
        added_edges_count += 1 
        if added_edges_count == int(k/2):
            print("Updating the best edges...")
            best_edges = add_new_edges(g)
            for edge, updated_dgap in best_edges:
                if edge == (s, t):
                    dgap = updated_dgap
                    break
        if added_edges_count >= k:
            break
    print(f"Edges deleted = {added_edges_count}")
    return g
