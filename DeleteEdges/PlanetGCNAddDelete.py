import warnings
warnings.filterwarnings('ignore')
import argparse
parser = argparse.ArgumentParser(description='Run BraessGCNPlanetoid script')
parser.add_argument('--filename', type=str, help='Dataset to download')
parser.add_argument('--out', type=str, help='name of log file')
parser.add_argument('--max_iters', type=int, default=10, help='maximum number of Braess iterations')
parser.add_argument('--dropout', type=float, default=0.4130296, help='Dropout = [Cora - 0.4130296, Citeseer - 0.3130296]')
parser.add_argument('--hidden_dimension', type=int, default=32, help='Hidden Dimension size')
parser.add_argument('--LR', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
parser.add_argument('--device',type=str,default='cpu')
args = parser.parse_args()


######### Hyperparams to use #############
## Cora --> Dropout = 0.4130296 ; LR = 0.01 ; Hidden_Dimension = 32
## Citeseer --> Dropout = 0.3130296 ; LR = 0.01 ; Hidden_Dimension = 32


import logging
logger = logging.getLogger()
fhandler = logging.FileHandler(filename= args.out, mode='a')
formatter = logging.Formatter('%(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)
import torch
import os
from torch_geometric.datasets import TUDataset, Planetoid,WebKB,Actor, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents
from torch_geometric.utils import to_networkx,from_networkx,homophily
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.nn import GCNConv
import torch
from torch.nn import Linear
import torch.nn.functional as F
from tqdm import tqdm
import networkx as nx
import numpy as np
import random
import pickle
from torch_geometric.nn import GCNConv
from scipy.sparse import coo_matrix,csr_matrix
import torch
from torch.nn import Linear
import torch.nn.functional as F
from tqdm import tqdm
from braess_rewire import braess_rewire
from braesstop import braess_pruning
device = torch.device(args.device)


def spectral_gap(G):
  Lmod = nx.normalized_laplacian_matrix(G).todense()
  valsmod,vecsmod = np.linalg.eigh(Lmod)
  return valsmod[1]

####Reproducibiltity and Benchmarking #####

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("==================================================================")
    print(f"Random seed set as {seed}")
 

val_seeds =   [3164711608, 3255599240, 894959334,  493067366,  3349051410,511641138,  2487307261, 951126382,  530590201,  17966177]

print(f"Downloading the dataset...")

print(f"")

##========================= Download Dataset ====================##
dataset = Planetoid(root = 'data/',name=args.filename,transform=NormalizeFeatures())
#dataset = Actor(root = 'data/',transform=NormalizeFeatures())
#dataset = WikipediaNetwork(root = 'data/',name=args.filename,transform=NormalizeFeatures())
print(f'Dataset: {dataset}:')
logging.info(f'Dataset: {dataset}')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
logging.info(f'Number of graphs: {len(dataset)}')
logging.info(f'Number of features: {dataset.num_features}')
logging.info(f'Number of classes: {dataset.num_classes}')
logging.info(f"======="*10)
data = dataset[0]  # Get the first graph object.
print("Done!")
print()
print(data)


# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

##========================= Calculate Homophilly Score ====================##

# before_score = homophily(data.edge_index, data.y, method='edge_insensitive')
# logging.info(f"Homophilly score before = {before_score}")
# logging.info(f"=========="*10)

##========================= Select the Largest Connected Component and Perform Edge Rewiring ====================##
print()
print(f"Selecting the LargestConnectedComponents..")
transform = LargestConnectedComponents()
data = transform(dataset[0])
print(data)
print()
print("Extracting the adjacency matrix for rewiring...")
nxgraph = to_networkx(data)
#gapbefore = spectral_gap(nxgraph)
#logging.info("Spectral Gap before = {gapafter}")
adj = nx.adjacency_matrix(nxgraph)
print()
max_iterations = args.max_iters
print(f"Performing Braess Minimization for {max_iterations} iterations")
logging.info(f'Deleting {max_iterations} edges')
adj = braess_pruning(adj,max_iterations)
#adj = braess_rewire(adj,max_iterations)
print("Done!")
print()
newg = nx.from_scipy_sparse_array(adj)
gapafter = spectral_gap(newg)
logging.info(f"Spectral Gap After = {gapafter}")
newdata = from_networkx(newg)
print(newdata)
data.edge_index = torch.cat([newdata.edge_index])
print()
print(data)

##========================= Split the dataset into train/test/val ====================##
print("Splitting datasets train/val/test...")
transform2 = RandomNodeSplit(split="train_rest",num_splits = 10, num_val=0.2, num_test=0.2)
data  = transform2(data)
print(data)
print()
print("Start Training...")
##========================= Training/Testing Initialization ====================##
p = args.dropout ### Dropout
lr = args.LR
data = data.to(device)
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=p, training=self.training)
        x = self.conv2(x, edge_index)
        return x


##========================= Training/Testing Loop ====================##
hidden_channel = [args.hidden_dimension] 
avg_acc = []
avg_acc_allsplits = []


for channels in range(len(hidden_channel)):
  print(f"Training for hidden_channel = {hidden_channel[channels]}")
  logging.info(f"Training for hidden_channel = {hidden_channel[channels]}")
  model = GCN(hidden_channel[channels])
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
  print(f"LR : {lr} Dropout : {p}")
  logging.info(f"LR : {lr} Dropout : {p}")
  criterion = torch.nn.CrossEntropyLoss()


  for split_idx in range(1,10):
      print(f"Training for index = {split_idx}")
      train_mask = data.train_mask[:,split_idx]
      test_mask = data.test_mask[:,split_idx]
      val_mask = data.val_mask[:,split_idx]

      def train():
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

      def val():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability. 
        val_correct = pred[val_mask] == data.y[val_mask]  # Check against ground-truth labels.
        val_acc = int(val_correct.sum()) / int(val_mask.sum())  # Derive ratio of correct predictions.
        return val_acc


      def test():
            model.eval()
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)  # Use the class with highest probability. 
            test_correct = pred[test_mask] == data.y[test_mask]  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / int(test_mask.sum())  # Derive ratio of correct predictions.
            return test_acc


      for seeds in val_seeds:
                set_seed(seeds)
                print("Start training ....")
                for epoch in tqdm(range(1, 101)):
                          loss = train()
                print()
                val_acc = val()
                test_acc = test()
                avg_acc.append(test_acc*100)
                print(f'Val Accuracy : {val_acc:.2f}, Test Accuracy: {test_acc:.2f} for seed',seeds)
                print()
      print(f'Final test accuracy of all seeds {np.mean(avg_acc):.2f} \u00B1 {np.std(avg_acc):.2f}' ) 
      avg_acc_allsplits.append(np.mean(avg_acc))
  print(f'Final test accuracy of all splits {np.mean(avg_acc_allsplits):.2f} \u00B1 {np.std(avg_acc_allsplits):.2f}')
  logging.info((f'Final test accuracy of all splits {np.mean(avg_acc_allsplits):.2f} \u00B1 {np.std(avg_acc_allsplits):.2f}'))
  logging.info(f'')
logging.info(f"======="*10)
