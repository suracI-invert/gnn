from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero, SAGEConv
from torch_geometric.transforms import ToUndirected
from dataset import load_data
from model import Model
import torch
import numpy as np
from sklearn.metrics import f1_score

class Net(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = SAGEConv(12, 12)
        self.conv2 = SAGEConv(12, 12)
    def forward(self, x):
        return self.conv1(x)

node_types = ['paper', 'author']
edge_types = [
    ('paper', 'cites', 'paper'),
    ('paper', 'written_by', 'author'),
    ('author', 'writes', 'paper'),
]
metadata = (node_types, edge_types)

net = Net()
net = to_hetero(net, metadata)