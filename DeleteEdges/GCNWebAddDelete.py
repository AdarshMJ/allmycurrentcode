

###### Hyperparams #####
# Cornell = 0.4130296,0.001, 128
# Wisconsin = 0.5130296, 0.001,128
# Texas = 0.4130296,0.001,128
# Actor = 0.2130296,0.01,128
# ChameleonDirected = 0.3130296,0.001, 128
# ChameleonFiltered = 0.2130296,0.01,128
# ChameleonFilteredDirected = 0.4130296,0.01,128
# SquirrelDirected = 0.5130296,0.01, 128
# SquirrelFiltered = 0.5130296,0.01,128
# SquirrelFilteredDirected = 0.2130296,0.01,128
########################################



import argparse
parser = argparse.ArgumentParser(description='Run BraessGCNWebKB script')
parser.add_argument('--path', type=str, default='', help='path to directory containing npz file')
parser.add_argument('--filename', type=str, help='name of npz file')
parser.add_argument('--out', type=str, help='name of log file')
parser.add_argument('--max_iters', type=int, default=10, help='maximum number of Braess iterations')
parser.add_argument('--dropout', type=float, default=0.4130296, help='maximum number of Braess iterations')
parser.add_argument('--hidden_dimension', type=int, default=128, help='maximum number of Braess iterations')
parser.add_argument('--lr', type=float, default=0.001, help='maximum number of Braess iterations')
parser.add_argument('--device',type=str,default='cpu')
args = parser.parse_args()
import logging
logging.disable(logging.DEBUG)
logger = logging.getLogger()
fhandler = logging.FileHandler(filename= args.out, mode='a')
formatter = logging.Formatter('%(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)
import torch
import warnings
warnings.filterwarnings('ignore')


import os
from torch_geometric.datasets import TUDataset, Planetoid,WebKB,Actor, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents
from torch_geometric.utils import to_networkx,from_networkx,to_scipy_sparse_matrix,from_scipy_sparse_matrix
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
from tqdm import tqdm
from scipy.sparse import coo_matrix,csr_matrix
device = torch.device(args.device)
import torch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.nn import GCNConv
import torch
import scipy.sparse as sp
from torch.nn import Linear
import torch.nn.functional as F
from braess_rewire import braess_rewire
from braesstop import braess_pruning



def spectral_gap(G):
  Lmod = nx.normalized_laplacian_matrix(G).todense()
  valsmod,vecsmod = np.linalg.eigh(Lmod)
  return valsmod[1]


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
 

val_seeds = [3164711608,894959334,2487307261,3349051410,493067366]

print("Download datasets from - https://github.com/yandex-research/heterophilous-graphs.git' ")
print()
print("Loading datasets as npz-file..")
filepath = os.path.join(args.path, args.filename)
#data = np.load('heterophilous-graphs/data/cornell.npz')
logging.info(f"Dataset = {args.filename}")
data = np.load(filepath)

x = torch.tensor(data['node_features'], dtype=torch.float)
y = torch.tensor(data['node_labels'], dtype=torch.long)
edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
num_classes = len(torch.unique(y))
num_targets = 1 if num_classes == 2 else num_classes
data = Data(x=x, edge_index=edge_index)

############ Braess Criterion #######################
print("Extracting the adjacency matrix for rewiring...")
#nxgraph = to_networkx(data,to_undirected=True)
nxgraph = to_networkx(data)
adj = nx.adjacency_matrix(nxgraph)
print()
max_iterations = args.max_iters
logging.info(f"Performing Braess rewiring for {max_iterations} iterations")
newadj = braess_pruning(adj,max_iterations)

##### Uncomment this for doing both addition and deletion of edges ######
#adj = braess_rewire(adj,max_iterations)
print("Done!")
print()

newg = nx.from_scipy_sparse_array(newadj)
gcc = max(nx.connected_components(newg), key=len)
giantC = newg.subgraph(gcc)
gapafter = spectral_gap(nx.Graph(giantC))
logging.info(f"Spectral Gap After = {gapafter}")
logging.info("=============================================")
newdata = from_networkx(newg)
print(newdata)
logging.info(newdata)
data.edge_index = torch.cat([newdata.edge_index])
########################################################################
print("Converting to PyG dataset...")
data.y = y
data.num_classes = num_classes
data.num_targets = num_targets
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask
print("Done!..")
print()
print(data)

logging.info(data)
logging.info(f'Number of nodes: {data.num_nodes}')
logging.info(f'Number of edges: {data.num_edges}')
logging.info(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
logging.info(f'Has isolated nodes: {data.has_isolated_nodes()}')
logging.info(f'Has self-loops: {data.has_self_loops()}')
logging.info(f'Is undirected: {data.is_undirected()}')
logging.info(f'Number of graphs: {len(data)}')
logging.info(f'Number of features: {data.num_features}')
logging.info(f'Number of classes: {data.num_classes}')
logging.info(f"")
logging.info(f"=========="*10)


# logging.info(f"Braess Iterations = {max_iterations}")
# logging.info(f"=========="*10)
# for j in tqdm(range((max_iterations))):
#       edge_index,_ = braess.edge_rewire(data.edge_index.numpy(), num_iterations=1)      
#       data.edge_index = torch.tensor(edge_index)
# data.edge_index = torch.cat([data.edge_index])

print("Done!")
print()
print()
print("Splitting datasets into train/test/val...")
transform2 = RandomNodeSplit(split="train_rest",num_splits = 10, num_val=0.2, num_test=0.2)
data  = transform2(data)
print(data)
print()
print("=============="*20)

print("Start training...")
p = args.dropout ### start here 
lr = args.lr
data = data.to(device)
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(data.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, data.num_classes)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=p, training=self.training)
        x = self.conv2(x, edge_index)
        return x

hidden_channel = [args.hidden_dimension]
avg_acc = []
avg_acc_allsplits = []
max_acc = []

for channels in range(len(hidden_channel)):
  print(f"Training for hidden_channel = {hidden_channel[channels]}")
  logging.info(f"Training for hidden_channel = {hidden_channel[channels]}")  
  model = GCN(hidden_channel[channels])
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
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
      print(f'Test Accuracy of all seeds {np.mean(avg_acc):.4f} \u00B1 {np.std(avg_acc):.4f}' ) 
      avg_acc_allsplits.append(np.mean(avg_acc))
  print(f'Final Test Accuracy of all splits {np.mean(avg_acc_allsplits):.4f} \u00B1 {np.std(avg_acc_allsplits):.4f}')
  logging.info((f'Final Test Accuracy of all splits {np.mean(avg_acc_allsplits):.4f} \u00B1 {np.std(avg_acc_allsplits):.4f}'))
  logging.info(f'')
logging.info(f"======="*10)