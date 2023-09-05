import torch
from dataset import load_data
from torch_geometric.transforms import ToUndirected
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
from model import Model
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def compute_metrics(logits, label):
    logits = logits.detach().cpu()
    predictions = np.zeros(logits.shape)
    predictions[np.where(logits.sigmoid() >= 0.5)] = 1
    label = label.detach().cpu().numpy()
    return {
        'accuracy': accuracy_score(label, predictions),
        'f1': f1_score(label, predictions, average= 'micro'),
        'roc_auc': roc_auc_score(label, predictions, average= 'micro')
    }


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def get_loaders():
    train_data = load_data('train')
    val_data = load_data('val')
    test_data = load_data('test')
    num_nodes_dict = {
        'cases': train_data['cases'].num_nodes + val_data['cases'].num_nodes + test_data['cases'].num_nodes,
        'applicants': train_data['applicants'].num_nodes + val_data['applicants'].num_nodes + test_data['applicants'].num_nodes,
        'defendants': train_data['defendants'].num_nodes + val_data['defendants'].num_nodes + test_data['defendants'].num_nodes
    }
    train_data = ToUndirected()(train_data)
    val_data = ToUndirected()(val_data)
    test_data = ToUndirected()(test_data)
    metadata = train_data.metadata()
    train_loader = NeighborLoader(train_data, [30] * 2, input_nodes= 'cases', batch_size= 128, shuffle= True)
    val_loader = NeighborLoader(val_data, [30] * 2, input_nodes= 'cases', batch_size= 128, shuffle= False)
    test_loader = NeighborLoader(test_data, [30] * 2, input_nodes= 'cases', batch_size= 128, shuffle= False)
    return train_loader, val_loader, test_loader, metadata, num_nodes_dict, train_data.num_classes

train_loader, val_loader, test_loader, metadata, num_nodes_dict, num_classes = get_loaders()


model = Model(64, 384, num_nodes_dict, num_classes, metadata).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr= 0.0005)

def train(epoch, train_loader):
    model.train()
    total_loss = total_examples = 0
    for sample in tqdm(train_loader):
        optimizer.zero_grad()
        sample.to(device)
        pred = model(sample)['cases']
        label = sample['cases'].y.to(dtype= torch.float32)
        loss = F.binary_cross_entropy_with_logits(pred, label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) + pred.numel()
        total_examples += pred.numel()
    print(f'Epoch: {epoch}, Loss: {total_loss / total_examples:.4f}')

def eval(epoch, val_loader):
    model.eval()
    with torch.no_grad():
        preds = []
        ground_truths = []
        for sample in tqdm(val_loader):
            sample.to(device)
            logits = model(sample)['cases']
            preds.append(logits)
            ground_truths.append(sample['cases'].y)
        pred = torch.cat(preds, dim= 0)
        ground_truth = torch.cat(ground_truths, dim= 0)
        res = compute_metrics(pred, ground_truth)
        print(f'Epoch {epoch}: {res}')

def test(test_loader):
    model.eval()
    with torch.no_grad():
        preds = []
        ground_truths = []
        for sample in tqdm(test_loader):
            sample.to(device)
            logits = model(sample)['cases']
            preds.append(logits)
            ground_truths.append(sample['cases'].y)
        pred = torch.cat(preds, dim= 0)
        ground_truth = torch.cat(ground_truths, dim= 0)
        res = compute_metrics(pred, ground_truth)
        print(f'Test, accuracy: {res}')

print('Training')
for epoch in range(1000):
    train(epoch, train_loader)
    eval(epoch, val_loader)
print('Testing')
test(test_loader)