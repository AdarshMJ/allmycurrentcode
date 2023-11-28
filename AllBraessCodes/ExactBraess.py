import numpy as np
import networkx as nx
import random
import scipy.sparse as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from scipy import stats
eps = 1e-6
def exactbraess(adj,pratio):
  print("Loading adjacency matrix..")
  n = adj.shape[0]
  criterion = np.zeros((n, n))
  num_initial_edges = (adj != 0).sum() // 2
  num_deleted_edges = 0
  with tqdm(total=num_initial_edges) as pbar:  
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v] != 0:           
                du = adj[[u], :].sum()
                dv = adj[[v], :].sum()
                adj[u, v] = 0
                adj[v, u] = 0 
                deg = np.array(adj.sum(axis=1)).flatten()
                D_sqrt_inv = sp.diags(np.power(deg+eps, -0.5))
                L_norm = sp.eye(adj.shape[0]) - D_sqrt_inv @ adj @ D_sqrt_inv
                valsold, vecsold = np.linalg.eigh(L_norm.todense())  
                fu = vecsold[u,1]
                fv = vecsold[v,1]
                gap = valsold[1]
                largest_eigenvector = vecsold[:, np.argmax(newgap)]
                proj = np.dot(vecsold[1], largest_eigenvector)/(np.linalg.norm(largest_eigenvector+eps))
                cond1 = (proj**2)*gap + 2*(1-gap)
                cond11 = np.divide(np.subtract(np.sqrt(du+1), np.sqrt(du)), np.sqrt(du+1))*(fu**2) 
                cond12 = np.divide(np.subtract(np.sqrt(dv+1), np.sqrt(dv)), np.sqrt(dv+1))*(fv**2)
                cond2 = (2*fu*fv)/np.sqrt(du+1)*np.sqrt(dv+1)
                final = cond1*(cond11+cond12)
                pbar.update(1)
                pbar.set_description(f"Checking Braess condition for edge ({u}, {v})")
                adj[u, v] = 1
                adj[v, u] = 1
                if (final<cond2):
                      criterion[u, v] = np.real(cond2-final)
                      criterion[v, u] = criterion[u,v]
    criterion_flat = criterion.flatten()
    indices_flat = np.arange(criterion.size)
    sorted_indices = indices_flat[np.argsort(criterion_flat)[::-1]] 
    top_k_indices = sorted_indices[:2*pratio]    
    for idx in top_k_indices:
              i, j = np.unravel_index(idx, (n, n))
              criterion_value = criterion[i, j]
              print(criterion_value)
              adj[i, j] = 0
              adj[j, i] = 0
              adj.eliminate_zeros()
              num_deleted_edges += 1
              du = adj[[u], :].sum()
              dv = adj[[v], :].sum()
              deg = np.array(adj.sum(axis=1)).flatten()
              D_sqrt_inv = sp.diags(np.power(deg+eps, -0.5))
              L_norm = sp.eye(adj.shape[0]) - D_sqrt_inv @ adj @ D_sqrt_inv
              valsnew, vecsnew = np.linalg.eigh(L_norm.todense())  
    print(f"Number of edges deleted = {num_deleted_edges/2}")
    print()    
    return adj_copy