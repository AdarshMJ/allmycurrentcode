import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score as NMI
import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected,to_networkx,from_networkx
from tqdm import tqdm
import networkx as nx
import argparse
from listedproxydelete import process_and_update_edges
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(1234567)

parser = argparse.ArgumentParser(description='Run SBM')
parser.add_argument('--path', type=str, default='Dataset/SBM_30_0.3_0.03_1e-10.pt', help='path to directory containing npz file')
args = parser.parse_args()


def visualize(h, color,legend_labels,title, filename,random_state = 13):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(5,5))
    plt.xticks([])
    plt.yticks([])
    scatter = plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
        # Add legend if legend_labels are provided
    if legend_labels:
        legend = plt.legend(handles= scatter.legend_elements()[0], title="Classes", labels=legend_labels)
        plt.setp(legend.get_title(), fontsize='12')

    if title:
        plt.title(title, fontsize=14)
    plt.savefig(filename)


dataset,_ = torch.load(args.path)
x = dataset['x']
edge_index = dataset['edge_index']
y = dataset['y']
SBMdata = Data(x=x,edge_index=edge_index,y=y,num_classes=2)
print(SBMdata)
before = to_networkx(SBMdata, to_undirected=True)
pos = nx.kamada_kawai_layout(before)
node_colors = [SBMdata.y[node] for node in before.nodes]
nx.draw(before, pos=pos, with_labels=False, node_color=node_colors, cmap=plt.cm.rainbow)
plt.savefig("Plots/MaximizeGap/SBMBeforePruning30.jpg")

print()
print("Pruning the graph for maximizing the spectral gap...")
SBMdatanx = to_networkx(SBMdata, to_undirected =True)
proxykardeletegap,Gpr = process_and_update_edges(SBMdatanx)
print()


pos = nx.kamada_kawai_layout(Gpr)
node_colors = [SBMdata.y[node] for node in Gpr.nodes]
nx.draw(Gpr, pos=pos, with_labels=False, node_color=node_colors, cmap=plt.cm.rainbow)
plt.savefig("Plots/MaximizeGap/SBMaxallsame/SBMAfterPruningallsame.jpg")


print("Converting to PyG dataset")
newdata = from_networkx(Gpr)
newdata.x = SBMdata.x
newdata.y = SBMdata.y
newdata.num_classes = SBMdata.num_classes
print(newdata)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        #self.conv1 = GCNConv(newdata.num_features, hidden_channels)
        #self.conv2 = GCNConv(hidden_channels, newdata.num_classes)
        self.conv1 = GCNConv(newdata.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, newdata.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=128)
print(model)

print("Visualizing the node embeddings before training...")
model.eval()
out = model(newdata.x, newdata.edge_index)
visualize(out, color=newdata.y,legend_labels=["Class 1", "Class 2"],title='Embeddings before training',filename="Plots/MaximizeGap/SBMaxallsame/BeforeTraining+PruneMaxallsame.jpg")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
train_losses = []
nmi_values = []
def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(newdata.x, newdata.edge_index)  # Perform a single forward pass.

    # Use the CrossEntropyLoss directly without masking here
    loss = criterion(out, newdata.y)

    # Compute the gradients and update parameters
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    clust = model(newdata.x, newdata.edge_index)
    return NMI(clust.max(1)[1].cpu(), newdata.y.cpu())

for epoch in range(1, 1001):
    train_loss = train()
    nmi = test()
    train_losses.append(train_loss)
    nmi_values.append(nmi)
    print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, ' f'NMI: {nmi:.3f}')

# Plotting the training loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Plots/MaximizeGap/SBMaxallsame/LossSBMallsame.png")

# Plotting the NMI values
plt.figure(figsize=(10, 5))
plt.plot(nmi_values, label='NMI')
plt.title('NMI Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('NMI')
plt.legend()
plt.savefig("Plots/MaximizeGap/SBMaxallsame/TestNMISBMallsame.png")
print("After training GCN, visualizing node embeddings...")
model.eval()
out = model(newdata.x, newdata.edge_index)
visualize(out, color=newdata.y,legend_labels=["Class 1", "Class 2"],title=f'Embeddings after training - NMI: {nmi:.3f}',filename = 'Plots/MaximizeGap/SBMaxallsame/AfterTraining+PruneMaxallsame.jpg')