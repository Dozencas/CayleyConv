import torch
from torch_geometric.nn import ChebConv, global_mean_pool
from torch.optim import Adam
from torch_geometric.datasets import TUDataset
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader
class ChebNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, K):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(input_dim, hidden_dim, K)
        self.conv2 = ChebConv(hidden_dim, output_dim, K)
        self.relu = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        # 聚合节点到图级别
        x = global_mean_pool(x, batch)
        return torch.nn.functional.log_softmax(x, dim=1)

dataset = TUDataset(root='data/TUDataset', name='MUTAG')
train_dataset = dataset[:150]
test_dataset = dataset[150:]
data = dataset[0]
# 数据分割
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChebNet(input_dim=dataset.num_features, hidden_dim=64, output_dim=dataset.num_classes, K=3).to(device)
optimizer = Adam(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()

# 训练函数
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 测试函数
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        _, pred = model(data).max(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)
for epoch in range(10):
    loss = train()
    test_acc = test(test_loader)
    print(f'Epoch: {epoch}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')
