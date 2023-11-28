import numpy as np
import random
import scipy.sparse as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import warnings
warnings.filterwarnings('ignore')
eps = 1e-6

def optimisegap(adj):
    #print("Loading adjacency matrix..")
    n = adj.shape[0]
    #print("Done! Calculating spectral gap...")
    deg = np.array(adj.sum(axis=1)).flatten()
    D_sqrt_inv = sp.diags(np.power(deg+eps, -0.5))
    L_norm = sp.eye(adj.shape[0]) - D_sqrt_inv @ adj @ D_sqrt_inv
    valsold, vecsold = sp.linalg.eigsh(L_norm, k=2, sigma=0, which='LM')
    gap = np.real(valsold[1])
    missing_edges = (adj == 0).multiply(adj.T == 0)
    edges = missing_edges.nonzero()
    best_gap = 0
    best_edge = None
    for u, v in zip(edges[0], edges[1]):
            if (u!=v):
                  adj[u, v] = 1
                  adj[v, u] = 1  
                  L_norm[u, v] += 1 / np.sqrt(deg[u] * deg[v])
                  L_norm[v, u] += 1 / np.sqrt(deg[u] * deg[v])
                  delta_w = 1 + 2 * L_norm[u, v]
                  vals_est = gap + delta_w * ((vecsold[u, 1] - vecsold[v, 1])**2 - gap * (vecsold[u, 1]**2 + vecsold[v, 1]**2))   
                  new_gap = np.real(vals_est)
                  if (new_gap > best_gap):
                    best_edge = u, v    
                    best_gap = new_gap
                  adj[u, v] = 0
                  adj[v, u] = 0  
  
    if best_edge is None:
          print("No best edges to add")
    else :
        u, v = best_edge
        adj[u, v] = 1
        adj[v, u] = 1
        L_norm[u, v] += 1 / np.sqrt(deg[u] * deg[v])
        L_norm[v, u] += 1 / np.sqrt(deg[u] * deg[v])
        _, vecsnew = sp.linalg.eigsh(L_norm, k=2, sigma=0, which='LM', v0 = vecsold[:,1])
        delta_w = 1 + 2 * L_norm[u, v]
        vals_est = gap + delta_w * ((vecsnew[u, 1] - vecsnew[v, 1])**2 - new_gap * (vecsnew[u, 1]**2 + vecsnew[v, 1]**2))                    
            
    #print(f"Number of edges added - {edges_added}")
    return adj
