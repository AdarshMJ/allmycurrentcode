import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score as NMI
import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected,to_networkx,from_networkx
from torch_geometric.transforms import RandomNodeSplit
from sklearn.model_selection import train_test_split
import networkx as nx
from nodeli import *
from tqdm import tqdm
import argparse
import random
import numpy as np


random.seed(42)
np.random.seed(42)
torch.manual_seed(1234567)

parser = argparse.ArgumentParser(description='Run SBM')
parser.add_argument('--path', type=str, default='Dataset/', help='path to directory containing npz file')
parser.add_argument('--SBMPlotName', type=str, default='OriginalSBM', help='Saving the SBM plot')
parser.add_argument('--EmbeddingPlotBefore', type=str, default='OriginalPlotBefore', help='Saving the node embeddings before training')
parser.add_argument('--EmbeddingPlotAfter', type=str, default='OriginalPlotAfter', help='Saving the node embeddings after training')
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



def get_graph_and_labels_from_pyg_dataset(dataset):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(dataset.x)))
    graph.add_edges_from(dataset.edge_index.T.numpy())

    labels = dataset.y.numpy()

    return graph, labels


dataset,_ = torch.load(args.path)
x = dataset['x']
edge_index = dataset['edge_index']
y = dataset['y']
SBMdata = Data(x=x,edge_index=edge_index,y=y,num_classes=2)
print(SBMdata)
graph, labels = get_graph_and_labels_from_pyg_dataset(SBMdata)
print("Calculating Different Informativeness Measures...")
print(f'node label informativeness: {li_node(graph, labels):.2f}')
print(f'adjusted homophily: {h_adj(graph, labels):.2f}')
print(f'edge label informativeness: {li_edge(graph, labels):.2f}')
print("=============================================================")
print()

print("Performing Train/Test splits...")
transform2 = RandomNodeSplit(split="train_rest", num_test=20)
SBMdata  = transform2(SBMdata)
print(SBMdata)
print(f'Number of training nodes: {SBMdata.train_mask.sum()}')
print(f'Number of testing nodes: {SBMdata.test_mask.sum()}')
print(SBMdata.test_mask)
# print("Train Data:")
# print(train_data)
# print("\nTest Data:")
# print(test_data)

gsbm = to_networkx(SBMdata,to_undirected=True)
print(gsbm)
pos = nx.kamada_kawai_layout(gsbm)
node_colors = [SBMdata.y[node] for node in gsbm.nodes]
nx.draw(gsbm, pos=pos, with_labels=False, node_color=node_colors, cmap=plt.cm.rainbow)
plt.savefig("Plots/Original/SBMFlipy/OriginalSBMFlipy.jpg")



# class GCN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         torch.manual_seed(1234567)
#         #self.conv1 = GCNConv(newdata.num_features, hidden_channels)
#         #self.conv2 = GCNConv(hidden_channels, newdata.num_classes)
#         self.conv1 = GCNConv(train_data.num_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, SBMdata.num_classes)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x

# model = GCN(hidden_channels=512)
# print(model)


# train_losses = []

# print("Visualizing the node embeddings before training...")
# model.eval()
# out = model(train_data.x, train_data.edge_index)
# visualize(out, color=train_data.y,legend_labels=["Class 1", "Class 2"],title='Embeddings before training',filename="Plots/Original/SBMFlipy/BeforeTrainingSBMFlipy.jpg")

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# criterion = torch.nn.CrossEntropyLoss()


# train_losses = []
# nmi_values = []
# def train():
#     model.train()
#     optimizer.zero_grad()  # Clear gradients.
#     out = model(train_data.x, train_data.edge_index)  # Perform a single forward pass.

#     # Use the CrossEntropyLoss directly without masking here
#     loss = criterion(out, train_data.y)

#     # Compute the gradients and update parameters
#     loss.backward()
#     optimizer.step()

#     return loss.item()

# @torch.no_grad()
# def test():
#     model.eval()
#     clust = model(test_data.x, test_data.edge_index)
#     return NMI(clust.max(1)[1].cpu(), test_data.y.cpu())

# for epoch in range(1, 1001):
#     train_loss = train()
#     train_losses.append(train_loss)
# nmi = test()
# print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, ' f'NMI: {nmi:.3f}')

# # Plotting the training loss
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.title('Training Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig("Plots/MaximizeGap/SBMaxallsame/LossSBMallsame.png")

# # Plotting the NMI values
# # plt.figure(figsize=(10, 5))
# # plt.plot(nmi_values, label='NMI')
# # plt.title('NMI Over Epochs')
# # plt.xlabel('Epoch')
# # plt.ylabel('NMI')
# # plt.legend()
# # plt.savefig("Plots/MaximizeGap/SBMaxallsame/TestNMISBMallsame.png")
# print("After training GCN, visualizing node embeddings...")
# model.eval()
# out = model(test_data.x, test_data.edge_index)
# visualize(out, color=test_data.y,legend_labels=["Class 1", "Class 2"],title=f'Embeddings after training - NMI: {nmi:.3f}',filename = 'Plots/Original/SBMFlipy/Testsetemb.jpg')

# # Plotting the training loss
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.title('Training Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig("Plots/Original/SBMFlipy/LossSBMFlipy.png")

# # Plotting the NMI values
# # plt.figure(figsize=(10, 5))
# # plt.plot(test_values, label='Test Accuarcy')
# # plt.title('Test Accuracy Over Epochs')
# # plt.xlabel('Epoch')
# # plt.ylabel('Test Acc')
# # plt.legend()
# # plt.savefig("Plots/Original/SBMFlipy/TestNMISBMFlipy.png")