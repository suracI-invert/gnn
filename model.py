import torch
from torch_geometric.nn import GATv2Conv, to_hetero, SAGEConv, HeteroDictLinear, Linear, Sequential
from torch_geometric.nn.models import GAT, GraphSAGE

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels) -> None:
        super().__init__()

        # self.conv1 = GATv2Conv((-1, -1), hidden_channels, heads= 6, dropout= 0.01, add_self_loops= False)
        # self.conv2 = GATv2Conv((-1, -1), hidden_channels, heads= 6, dropout= 0.01, add_self_loops= False)

        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        # self.model = GraphSAGE(hidden_channels, hidden_channels, 12, hidden_channels)
        # self.lin = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class Classifier(torch.nn.Module):
    def forward(self, x_cases, x_articles, edge_label_index):
        edge_feat_case = x_cases[edge_label_index[0]]
        edge_feat_article = x_articles[edge_label_index[1]]
        return (edge_feat_case * edge_feat_article).sum(dim= -1)
    
class Model(torch.nn.Module):
    def __init__(self, hidden_channels, embedding_dim, num_nodes, num_classes, data_metadata) -> None:
        super().__init__()
        # self.article_lin = torch.nn.Linear(embedding_dim, hidden_channels)
        self.app_lin = torch.nn.Linear(embedding_dim, hidden_channels)
        self.def_lin = torch.nn.Linear(embedding_dim, hidden_channels)
        self.cases_emb = torch.nn.Embedding(num_nodes['cases'], hidden_channels)
        self.app_emb = torch.nn.Embedding(num_nodes['applicants'], hidden_channels)
        self.def_emb = torch.nn.Embedding(num_nodes['defendants'], hidden_channels)
        # self.article_emb = torch.nn.Embedding(num_nodes['articles'], hidden_channels)
        # self.gnn = GNN(hidden_channels)
        self.gnn = Sequential('x, edge_index', [
            (SAGEConv(hidden_channels, hidden_channels), 'x, edge_index -> x'), torch.nn.ReLU(True),
            (SAGEConv(hidden_channels, hidden_channels), 'x, edge_index -> x'), torch.nn.ReLU(True),
            (SAGEConv(hidden_channels, hidden_channels), 'x, edge_index -> x'), torch.nn.ReLU(True),
            (SAGEConv(hidden_channels, hidden_channels), 'x, edge_index -> x'), torch.nn.ReLU(True),
            (SAGEConv(hidden_channels, hidden_channels), 'x, edge_index -> x'), torch.nn.ReLU(True),
            (SAGEConv(hidden_channels, hidden_channels), 'x, edge_index -> x'), torch.nn.ReLU(True),
            (Linear(hidden_channels, num_classes), 'x -> x'),
        ])
        
        self.gnn = to_hetero(self.gnn, metadata= data_metadata)
        # self.classifier = Classifier()
        # self.lin = Linear(hidden_channels, num_classes)
        # self.lin = to_hetero(self.lin, data_metadata)

    def forward(self, data):
        x_dict = {
            'cases': self.cases_emb(data['cases'].node_id),
            'applicants': self.app_lin(data['applicants'].x) + self.app_emb(data['applicants'].node_id),
            'defendants': self.def_lin(data['defendants'].x) + self.def_emb(data['defendants'].node_id),
            # 'articles': self.article_lin(data['articles'].x) + self.article_emb(data['articles'].node_id)
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        # pred = self.classifier(
        #     x_dict['cases'], x_dict['articles'], data['cases', 'violates', 'articles'].edge_label_index
        # )
        # x = self.lin(x_dict)
        return x_dict