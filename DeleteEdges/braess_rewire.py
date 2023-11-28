import numpy as np
import random
import scipy.sparse as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse.linalg
eps = 1e-6

def braess_rewire(adj,pratio):
    print("Loading adjacency matrix..")
    n = adj.shape[0]
    adj_copy = adj
    print("Done! Calculating spectral gap...")
    deg = np.array(adj_copy.sum(axis=1)).flatten()
    deg = deg+eps
    D_sqrt_inv = sp.diags(np.power(deg, -0.5))
    L_norm = sp.eye(adj_copy.shape[0]) - D_sqrt_inv @ adj_copy @ D_sqrt_inv
    valsold, vecsold = sp.linalg.eigs(L_norm, k=2, sigma=0, which='LM') # comput
    valsold = np.real(valsold)
    print("Done!...")
    missing_edges = (adj_copy == 0).multiply(adj_copy.T == 0) # find all missing edges that are not self-loops
    edges = missing_edges.nonzero()
    candidates = min(pratio, len(edges[0]))
    edges_added = 0
    with tqdm(total=candidates) as pbar:  
        for u, v in zip(edges[0][:pratio], edges[1][:pratio]):
              proj = 1.0
              du = adj_copy[[u], :].sum()
              dv = adj_copy[[v], :].sum()
              fu = vecsold[u,1]
              fv = vecsold[v,1]
              cond1 = (proj**2)*valsold[1] + 2*(1-valsold[1])
              cond11 = np.divide(np.subtract(np.sqrt(du+1), np.sqrt(du)), np.sqrt(du+1)) 
              cond12 = np.divide(np.subtract(np.sqrt(dv+1), np.sqrt(dv)), np.sqrt(dv+1))
              cond2 = (2*fu*fv)/np.sqrt(du+1)*np.sqrt(dv+1)
              final = cond1*(cond11+cond12)
              pbar.update(1)
              pbar.set_description(f"Checking Braess condition for edge ({u}, {v})")
              if (final > cond2) and (u != v) and (adj_copy[u,v] == 0) :
                  adj_copy[u, v] = 1
                  adj_copy[v, u] = 1    
                  du = adj_copy[[u]].sum() 
                  dv = adj_copy[[v]].sum()
                  edges_added +=1
                  deg = np.array(adj_copy.sum(axis=1)).flatten()
                  D_sqrt_inv = sp.diags(np.power(deg, -0.5))
                  L = sp.eye(adj_copy.shape[0]) - D_sqrt_inv @ adj_copy @ D_sqrt_inv
                  delta_w = 1 +2 * L[u, v]
                  valsold = valsold + delta_w * ((vecsold[u,1] - vecsold[v,1])**2 
                                                          - valsold * (vecsold[u,1] ** 2 + vecsold[v,1] ** 2))
                  vecsnow = ((((vecsold[u,1]).T*L*(vecsold[v,1]))/(valsold[1]))*(vecsold[u,1])).toarray()
                  largest_eigenvector = vecsnow[:, np.argmax(valsold)]
                  proj = np.dot(vecsnow[1], largest_eigenvector)/(np.linalg.norm(largest_eigenvector+eps))
                  fu=fu+vecsnow[u,1]
                  fv=fv+vecsnow[v,1]     
    #               
    print(f"Number of edges added - {edges_added}")
    return adj_copy
