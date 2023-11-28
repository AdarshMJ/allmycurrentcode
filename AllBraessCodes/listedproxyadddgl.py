import networkx as nx
import scipy.sparse as sp
import numpy as np
import warnings
import dgl

warnings.filterwarnings('ignore')
eps = 1e-6

def proxy_add(missing_edges, deg,vecsold, gap):
    deg = deg[:, np.newaxis] 
    vecsold = np.divide(vecsold, np.sqrt(deg))
    vecsold = vecsold / np.linalg.norm(vecsold, axis=1)[:, np.newaxis]
    delta_gap = (vecsold[missing_edges[0],1]-vecsold[missing_edges[1],1])**2-gap*(vecsold[missing_edges[0],1]**2 + vecsold[missing_edges[1],1]**2)
    index_best_edge = np.argmax(delta_gap)
    return missing_edges[0][index_best_edge],missing_edges[1][index_best_edge],index_best_edge

# def proxy_add(i,j, deg,vecsold, gap):
#     deg = deg[:, np.newaxis] 
#     vecsold = np.divide(vecsold, np.sqrt(deg))
#     vecsold = vecsold / np.linalg.norm(vecsold, axis=1)[:, np.newaxis]
#     delta_gap = (vecsold[i,1]-vecsold[j,1])**2-gap*(vecsold[i,1]**2 + vecsold[j,1]**2)
#     index_best_edge = np.argmax(delta_gap)
#     return [i][index_best_edge],[j][index_best_edge],index_best_edge

def return_missing_edges(G):
    adj = nx.adjacency_matrix(G)
    edges = (adj == 0).multiply(adj.T == 0)
    missing_edges = sp.triu(edges,k=1)
    candidates = missing_edges.nonzero() 
    i, j = candidates[0], candidates[1]
    return i, j

# def return_missing_edges(G):
#   gc = nx.complement(G)
#   missing_edges = list(gc.edges) 
#   return missing_edges

def obtain_Lnorm(G):
    adj = nx.adjacency_matrix(G)
    self_loops = sp.diags(np.ones(adj.shape[0]))
    deg = np.array(adj.sum(axis=1)).flatten()   
    D_sqrt_inv = sp.diags(np.power(deg, -0.5))
    #L_norm = sp.eye(adj.shape[0]) - D_sqrt_inv @ adj @ D_sqrt_inv
    L_norm = sp.eye(adj.shape[0]) - D_sqrt_inv @ adj @ D_sqrt_inv + self_loops
    return deg,L_norm

# def spectral_gap(L_norm):
#     gap, vectors = sp.linalg.eigsh(L_norm, k=2, sigma=0, which='LM')
#     return gap[1],vectors

def spectral_gap(L_norm):
    try:
        # Try the first approach
        gap, vectors = sp.linalg.eigsh(L_norm, k=2, sigma=0, which='LM')
        return gap[1], vectors
    except sp.linalg.ArpackNoConvergence:
        # If the first approach fails, try the second approach
        # Add a small regularization term to the matrix
        L_norm_regularized = L_norm + np.eye(L_norm.shape[0]) * 1e-6
        gap, vectors = sp.linalg.eigsh(L_norm_regularized, k=2, sigma=0, which='LM')
        return gap[1], vectors


def update_gap(L_norm,vecsold):
    _, vecsnew = sp.linalg.eigsh(L_norm, k=2, sigma=0, which='LM',v0 = vecsold[:,1])
    return vecsnew

def update_Lnorm(u,v,L_norm,deg):
      L_norm[u, :] *= np.sqrt(deg[u] / (deg[u] + 1))
      L_norm[:, u] = L_norm[u, :].T
      L_norm[v,:] *= np.sqrt(deg[v]/(deg[v] + 1))
      L_norm[:,v] = L_norm[v,:].T
      L_norm[u,v] = -1/np.sqrt((deg[u]+1)*(deg[v]+1))
      deg[u]+=1
      deg[v]+=1
      return deg,L_norm

def add_best_edges(g,edge_holder):
    adj_mod = nx.adjacency_matrix(g)
    for edge in edge_holder:
          adj_mod[edge[0], edge[1]] = 1
          adj_mod[edge[1], edge[0]] = 1
    newg = nx.from_scipy_sparse_array(adj_mod)
    return dgl.from_networkx(newg)
