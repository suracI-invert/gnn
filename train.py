from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from dataset import load_data
from model import Model
from tqdm import  tqdm
import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score, accuracy_score

data = load_data('train')
data = ToUndirected()(data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(data)

transform = RandomLinkSplit(
    num_val= 0.1,
    num_test= 0.1,
    disjoint_train_ratio= 0.3,
    neg_sampling_ratio= 2.0,
    add_negative_train_samples= False,
    edge_types= ('cases', 'violates', 'articles'),
    rev_edge_types= ('articles', 'rev_violates', 'cases')
)
train_data, val_data, test_data = transform(data)

train_loader = LinkNeighborLoader(
    data= train_data,
    num_neighbors = [20, 10],
    neg_sampling_ratio= 2.0,
    edge_label_index= (('cases', 'violates', 'articles'), train_data['cases', 'violates', 'articles'].edge_label_index),
    edge_label= train_data['cases', 'violates', 'articles'].edge_label,
    batch_size= 128,
    shuffle= True
)

val_loader = LinkNeighborLoader(
    data= val_data,
    num_neighbors= [20, 10],
    edge_label_index= (('cases', 'violates', 'articles'), val_data['cases', 'violates', 'articles'].edge_label_index),
    edge_label= val_data['cases', 'violates', 'articles'].edge_label,
    batch_size = 128,
    shuffle= False
)

test_loader = LinkNeighborLoader(
    data= test_data,
    num_neighbors= [20, 10],
    edge_label_index= (('cases', 'violates', 'articles'), test_data['cases', 'violates', 'articles'].edge_label_index),
    edge_label= test_data['cases', 'violates', 'articles'].edge_label,
    batch_size = 128,
    shuffle= False
)


model = Model(64, 384, {'cases': data['cases'].num_nodes, 'applicants': data['applicants'].num_nodes, 'defendants': data['defendants'].num_nodes, 'articles': data['articles'].num_nodes}, data.metadata())

model = model.to(device= device)

optimizer = torch.optim.Adam(model.parameters(), lr= 0.01)

def train(epoch):
    model.train()
    total_loss = total_examples = 0
    for sample in tqdm(train_loader):
        optimizer.zero_grad()
        sample.to(device)
        pred = model(sample)
        ground_truth = sample['cases', 'violates', 'articles'].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) + pred.numel()
        total_examples += pred.numel()
    print(f'Epoch: {epoch}, Loss: {total_loss / total_examples:.4f}')

# test_sample = next(iter(train_loader)).to(device)
# print(model(test_sample).softmax(dim= -1) > 0.5)

def eval(epoch):
    model.eval()
    preds = []
    ground_truths = []
    with torch.no_grad():
        for sample in tqdm(val_loader):
            sample.to(device)
            pred = model(sample)
            preds.append(pred.softmax(dim= -1) > 0.5)
            ground_truths.append(sample['cases', 'violates', 'articles'].edge_label)
        pred = torch.cat(preds, dim= 0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim= 0).cpu().numpy()
        acc = accuracy_score(ground_truth, pred)
        score = f1_score(ground_truth, pred, average= 'micro')
        print(f'Epoch: {epoch}, accuracy: {acc:.4f}, f1: {score:.4f}')

def test():
    model.eval()
    preds = []
    ground_truths = []
    with torch.no_grad():
        for sample in tqdm(test_loader):
            sample.to(device)
            pred = model(sample)
            preds.append(pred.softmax(dim= -1) > 0.5)
            ground_truths.append(sample['cases', 'violates', 'articles'].edge_label)
        pred = torch.cat(preds, dim= 0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim= 0).cpu().numpy()
        acc = accuracy_score(ground_truth, pred)
        score = f1_score(ground_truth, pred, average= 'micro')
        print(f'Test, accuracy: {acc:.4f}, f1: {score:.4f}')

for epoch in range(1000):
    print('Training')
    train(epoch)
    print("Evaluating")
    eval(epoch)
test()