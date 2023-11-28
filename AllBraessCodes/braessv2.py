import numpy as np
import random
import scipy.sparse as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse.linalg
eps = 1e-6

sparsity_holder = []
def braess_pruning(adj_copy,pratio):
  print("Loading adjacency matrix..")
  n = adj_copy.shape[0]
  print("Done! Calculating spectral gap...") 
  deg = np.array(adj_copy.sum(axis=1)).flatten()
  D_sqrt_inv = sp.diags(np.power(deg+eps, -0.5))
  L_norm = sp.eye(adj_copy.shape[0]) - D_sqrt_inv @ adj_copy @ D_sqrt_inv
  valsold, vecsold = sp.linalg.eigs(L_norm, k=2,sigma=0, which='LM') # compute smallest eigenvalues and eigenvectors
  gap = np.real(valsold[1])
  #print(f"Spectral Gap Before = {gap}")
  print("Done!...")
  criterion = np.zeros((n, n))
  num_initial_edges = (adj_copy != 0).sum() // 2
  num_deleted_edges = 0
  with tqdm(total=num_initial_edges) as pbar:  
    for i in range(n):
        for j in range(i+1, n):
            if adj_copy[i, j] != 0:
                proj = 1.0
                du = adj_copy[[i], :].sum()
                dv = adj_copy[[j], :].sum()
                fu = vecsold[i,1]
                fv = vecsold[j,1]
                cond1 = (proj**2)*gap + 2*(1-gap)
                cond11 = np.divide(np.subtract(np.sqrt(du+1), np.sqrt(du)), np.sqrt(du+1))*(fu**2) 
                cond12 = np.divide(np.subtract(np.sqrt(dv+1), np.sqrt(dv)), np.sqrt(dv+1))*(fv**2)
                cond2 = (2*fu*fv)/np.sqrt(du+1)*np.sqrt(dv+1)
                final = cond1*(cond11+cond12)
                pbar.update(1)
                pbar.set_description(f"Checking Braess condition for edge ({i}, {j})")
                criterion[i, j] = (final<cond2)
                criterion[j, i] = criterion[i, j]                  
    edges = np.argwhere(adj_copy)
    edge_scores = criterion[edges[:, 0], edges[:, 1]]
    sorted_indices = np.argsort(-edge_scores)
    top_edge_index = sorted_indices[0]
    u, v = edges[top_edge_index]
    adj_copy[u, v] = 0
    adj_copy[v, u] = 0
    adj_copy.eliminate_zeros()
    num_deleted_edges += 1
    for i in sorted_indices[:pratio-1]:
        u, v = edges[i]
        du = adj_copy[[u]].sum()
        dv = adj_copy[[v]].sum()
        deg = np.array(adj_copy.sum(axis=1)).flatten()
        D_sqrt_inv = sp.diags(np.power(deg+eps, -0.5))
        L_norm = sp.eye(adj_copy.shape[0]) - D_sqrt_inv @ adj_copy @ D_sqrt_inv
        delta_w = 1 - 2 * L_norm[u, v]
        gap = gap+delta_w * ((vecsold[u,1] - vecsold[v,1])**2 
                                                - gap * (vecsold[u,1] ** 2 + vecsold[v,1] ** 2))
        vecsnow = ((((vecsold[u,1]).T*L_norm*(vecsold[v,1]))/(valsold[1]))*(vecsold[u,1])).toarray()
        largest_eigenvector = vecsnow[:, np.argmax(gap)]
        proj = np.dot(vecsnow[1], largest_eigenvector)/(np.linalg.norm(largest_eigenvector+eps))
        fu=fu+vecsnow[u,1]
        fv=fv+vecsnow[v,1]   
        cond1 = (proj**2)*gap + 2*(1-gap)
        cond11 = np.divide(np.subtract(np.sqrt(du+1), np.sqrt(du)), np.sqrt(du+1))*(fu**2) 
        cond12 = np.divide(np.subtract(np.sqrt(dv+1), np.sqrt(dv)), np.sqrt(dv+1))*(fv**2)
        cond2 = (2*fu*fv)/np.sqrt(du+1)*np.sqrt(dv+1)
        final = cond1*(cond11+cond12)
        criterion[u, v] = (final<cond2)
        criterion[v, u] = criterion[u, v]                  
        edges = np.argwhere(adj_copy)
        edge_scores = criterion[edges[:, 0], edges[:, 1]]
        sorted_indices = np.argsort(-edge_scores)
        adj_copy[u, v] = 0
        adj_copy[v, u] = 0
        adj_copy.eliminate_zeros()  
        num_deleted_edges += 1
    print()
    print(f"Number of edges deleted = {num_deleted_edges}")
    print()    
    return adj_copy