import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from ast import literal_eval
from torch_geometric.data import HeteroData

def load_node_csv(path, index_col, encoders= None, **kwargs):
    df = pd.read_csv(path, **kwargs)
    mapping = {idx: i for i, idx in enumerate(df[index_col].unique())}

    x = None

    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim= -1)
    
    return x, mapping

def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders= None, labels= False, **kwargs):
    df = pd.read_csv(path, **kwargs)
    if labels:

        df = df[df['score'] > 1]
    src = [src_mapping[idx] for idx in df[src_index_col]]
    dst = [dst_mapping[idx] for idx in df[dst_index_col]]
    edge_idx = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim= -1)

    return edge_idx, edge_attr

class SequenceEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2', device= None):
        self.device = device
        self.model = SentenceTransformer(model_name, device= device)
    
    def __call__(self, df):
        with torch.no_grad():
            x = self.model.encode(df.values, show_progress_bar= True, convert_to_tensor= True, device= self.device)
        return x.cpu()
    
def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

class ScoreEncoder:
    def __init__(self):
        pass

    
    def __call__(self, df):
        min_val = min(df.values)
        max_val = max(df.values)
        x = torch.tensor([normalize(val, min_val, max_val) for val in df.values])
        return x

class MultiLabelEncoder:
    def __init__(self):
        pass

    def __call__(self, df):
        x = torch.tensor([literal_eval(df.values[i]) for i in range(len(df.values))])
        return x
    

def load_data(desc):
    _, cases_mapping = load_node_csv(f'./data/{desc}_data.csv', 'case_id')
    applicants_x, applicants_mapping = load_node_csv(f'./data/{desc}_applicants.csv', 'applicants', {'applicants': SequenceEncoder(device= 'cuda')})
    defendants_x, defendants_mapping = load_node_csv(f'./data/{desc}_defendants.csv', 'defendants', {'defendants': SequenceEncoder(device= 'cuda')})
    # article_x, article_mapping = load_node_csv('./data/articles.csv', 'id', {'article': SequenceEncoder(device= 'cuda')})
    y, _ = load_node_csv(f'./data/{desc}_allegedly_violated_articles_labels.csv', 'case_id', {'labels': MultiLabelEncoder()})

    cases_applicants_idx, _ = load_edge_csv(f'./data/{desc}_applicant_map.csv', 'case_id', cases_mapping, 'applicants', applicants_mapping)
    cases_defendants_idx, _ = load_edge_csv(f'./data/{desc}_defendant_map.csv', 'case_id', cases_mapping, 'defendants', defendants_mapping)
    cases_cases_idx, _ = load_edge_csv(f'./data/{desc}_data_mapping.csv', 'src', cases_mapping, 'dest', cases_mapping, labels= True)
    # cases_article_idx, _ = load_edge_csv(f'./data/{desc}_allegedly_violated_articles.csv', 'case_id', cases_mapping, 'article_id', article_mapping)

    data = HeteroData()
    data['cases'].node_id = torch.tensor(list(cases_mapping.values()), dtype= torch.int32)
    data['applicants'].node_id = torch.tensor(list(applicants_mapping.values()), dtype= torch.int32)
    data['defendants'].node_id = torch.tensor(list(defendants_mapping.values()), dtype= torch.int32)
    # data['articles'].node_id = torch.tensor(list(article_mapping.values()), dtype= torch.int32)
    # data['cases'].x = torch.ones((len(cases_mapping), 384))
    data['cases'].y = y
    data.num_classes = y.shape[1]
    data['applicants'].x = applicants_x
    data['defendants'].x = defendants_x
    # data['articles'].x = article_x

    data['cases', 'has', 'defendants'].edge_index = cases_defendants_idx
    data['cases', 'has', 'applicants'].edge_index = cases_applicants_idx
    data['cases', 'to', 'cases'].edge_index = cases_cases_idx
    # data['cases', 'to', 'cases'].edge_attr = cases_cases_attr
    # data['cases', 'violates', 'articles'].edge_index = cases_article_idx

    return data