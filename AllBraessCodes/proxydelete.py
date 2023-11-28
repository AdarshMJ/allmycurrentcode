import numpy as np
import random
import scipy.sparse as sp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
eps = 1e-6
np.random.seed(42)


def optimisedelete(adj):
    n = adj.shape[0]
    deg = np.array(adj_copy.sum(axis=1)).flatten()
    D_sqrt_inv = sp.diags(np.power(deg+eps, -0.5))
    L_norm = sp.eye(adj.shape[0]) - D_sqrt_inv @ adj @ D_sqrt_inv
    valsold, vecsold = sp.linalg.eigs(L_norm, k=2, sigma=0, which='LM')
    gap = np.real(valsold[1])
    best_gap = 0
    best_edge = None
    for u in range(n):
        for v in range(u+1, n):
          if adj[u, v] != 0:   
                  adj[u, v] = 0
                  adj[v, u] = 0  
                  adj.eliminate_zeros()  
                  L_norm[u, v] -= 1 / np.sqrt(deg[u] * deg[v])
                  L_norm[v, u] -= 1 / np.sqrt(deg[u] * deg[v])
                  delta_w = 1 - 2 * L_norm[u, v]
                  vals_est = gap - delta_w * ((vecsold[u, 1] - vecsold[v, 1])**2 - gap * (vecsold[u, 1]**2 + vecsold[v, 1]**2))  
                  new_gap = vals_est
                  if (new_gap > best_gap):
                    best_edge = u, v    
                    best_gap = new_gap
                  adj[u, v] = 1
                  adj[v, u] = 1   
    
    if best_edge is None :
              print("No best edges to add")
    else :
            u, v = best_edge
            adj[u, v] = 0
            adj[v, u] = 0
            adj.eliminate_zeros()
            L_norm[u, v] -= 1 / np.sqrt(deg[u] * deg[v])
            L_norm[v, u] -= 1 / np.sqrt(deg[u] * deg[v])
            _, vecsnew = sp.linalg.eigsh(L_norm, k=2, sigma=0, which='LM',v0=vecsold[:,1])
            delta_w = 1 - 2 * L_norm[u, v]
            vals_est = gap - delta_w * ((vecsnew[u, 1] - vecsnew[v, 1])**2 - gap * (vecsnew[u, 1]**2 + vecsnew[v, 1]**2))  
    return adj
