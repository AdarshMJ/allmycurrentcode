import numpy as np
import networkx as nx
import random
import scipy.sparse as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from scipy import stats
eps = 1e-6
def braess_pruning(adj_copy,pratio):
  print("Loading adjacency matrix..")
  n = adj_copy.shape[0]
  print("Done! Calculating spectral gap...") 
  deg = np.array(adj_copy.sum(axis=1)).flatten()
  D_sqrt_inv = sp.diags(np.power(deg+eps, -0.5))
  L_norm = sp.eye(adj_copy.shape[0]) - D_sqrt_inv @ adj_copy @ D_sqrt_inv
  valsold, vecsold = np.linalg.eigh(L_norm.todense())
  #valsold, vecsold = sp.linalg.eigs(L_norm, k=n,sigma=0, which='LM') 
  gap = np.real(valsold[1])
  largest_eigenvector = vecsold[:, np.argmax(gap)]
  proj = np.dot(vecsold[1], largest_eigenvector)/(np.linalg.norm(largest_eigenvector+eps))
  print("Done!...")
  criterion = np.zeros((n, n))
  num_initial_edges = (adj_copy != 0).sum() // 2
  num_deleted_edges = 0
  with tqdm(total=num_initial_edges) as pbar:  
    for u in range(n):
        for v in range(u+1, n):
            if adj_copy[u, v] != 0:           
                du = adj_copy[[u], :].sum()
                dv = adj_copy[[v], :].sum()
                adj_copy[u, v] = 0
                adj_copy[v, u] = 0   
                delta_w = 1 - 2 * L_norm[u, v]
                newgap = gap+delta_w * ((vecsold[u,1] - vecsold[v,1])**2 
                                                - gap * (vecsold[u,1] ** 2 + vecsold[v,1] ** 2)) 
                vecsnow = ((((vecsold[u,1]).T*L_norm*(vecsold[v,1]))/(newgap))*(vecsold[u,1])).toarray()
                fu = vecsold[u,1]+vecsnow[u,1]
                fv = vecsold[v,1]+vecsnow[v,1]
                largest_eigenvector = vecsnow[:, np.argmax(newgap)]
                proj = np.dot(vecsold[1]+vecsnow[1], largest_eigenvector)/(np.linalg.norm(largest_eigenvector+eps))
                cond1 = (proj**2)*newgap + 2*(1-newgap)
                cond11 = np.divide(np.subtract(np.sqrt(du+1), np.sqrt(du)), np.sqrt(du+1))*(fu**2) 
                cond12 = np.divide(np.subtract(np.sqrt(dv+1), np.sqrt(dv)), np.sqrt(dv+1))*(fv**2)
                cond2 = (2*fu*fv)/np.sqrt(du+1)*np.sqrt(dv+1)
                final = cond1*(cond11+cond12)
                pbar.update(1)
                pbar.set_description(f"Checking Braess condition for edge ({u}, {v})")
                adj_copy[u, v] = 1
                adj_copy[v, u] = 1
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
              adj_copy[i, j] = 0
              adj_copy[j, i] = 0
              adj_copy.eliminate_zeros()
              num_deleted_edges += 1
              du = adj_copy[[u], :].sum()
              dv = adj_copy[[v], :].sum()
              delta_w = 1 - 2 * L_norm[i, j]
              newgap = newgap+delta_w * ((vecsnow[i,1] - vecsnow[j,1])**2 
                                          - newgap * (vecsnow[i,1] ** 2 + vecsnow[j,1] ** 2)) 
              vecsnow = ((((vecsnow[i,1]).T*L_norm*(vecsnow[j,1]))/(newgap))*(vecsnow[i,1])).toarray()
              fu = fu+vecsnow[i,1]
              fv = fv+vecsnow[j,1]
              largest_eigenvector = vecsnow[:, np.argmax(newgap)]
              proj = np.dot(vecsnow[1], largest_eigenvector)/(np.linalg.norm(largest_eigenvector+eps))

    print(f"Number of edges deleted = {num_deleted_edges/2}")
    print()    
    return adj_copy