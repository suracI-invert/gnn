from dataset import load_node_csv
import torch

x, mapping = load_node_csv('./data/articles.csv', 'id')

print(mapping.values())
# print(torch.tensor(mapping.values(), dtype= torch.int16))