import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
from torch import Tensor
from torch.nn import Parameter
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix

from torch_geometric.nn import MessagePassing, TopKPooling, global_mean_pool, ChebConv
from torch_geometric.utils import degree, get_laplacian
from tqdm import trange
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor

time_stamps = {
    'data_loading': [],
    'forward_pass': [],
    'loss_calculation': [],
    'backward_pass': [],
    'optimization': [],
    'epoch_time': []
}

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.cfloat)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight.real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight.imag, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class CayleyConv(MessagePassing):
    def __init__(
            self,
            r: int,
            K: int,
            in_channels: int,
            out_channels: int,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert r > 0
        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.r = r
        self.K = K
        self.h = Parameter(torch.tensor(0.5, dtype=torch.float))
        self.i = Parameter(torch.tensor(0. + 1.j), requires_grad=False)
        # C_0
        self.real_linear = nn.Linear(in_channels, out_channels, bias=False)
        # C_1...C_r
        self.complex_linears = nn.ModuleList([ComplexLinear(in_channels, out_channels, bias=False) for _ in range(r)])
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
    ) -> Tensor:

        y_j = x
        out = self.real_linear(y_j)

        n_nodes = x.size(self.node_dim)
        device = edge_index.device
        edge_weight = torch.ones(edge_index.size(1), device=device)
        # Laplacian
        l_index, l_weight = get_laplacian(edge_index, edge_weight, normalization=None, num_nodes=n_nodes)

        # Jacobi
        jacobi, b = self.calculate_jacobi_and_b(l_index, l_weight)

        # Calculate r polynomials
        for j in range(self.r):

            b_j = self.calculate_b(b, y_j)
            y_j_k = b_j

            # K Jacobi iterations
            for k in range(self.K):
                y_j_k = self.propagate(l_index, x=y_j_k, jacobi=jacobi) + b_j
            y_j = y_j_k
            out = out + 2 * (self.complex_linears[j](y_j)).real
        return out

    def message(self, x_j: Tensor, jacobi: Tensor) -> Tensor:
        return jacobi.view(-1, 1) * x_j

    def calculate_jacobi_and_b(self, l_index, l_weight):
        l_row, l_col = l_index

        # Compute (hD + iI)^-1
        l_dia = l_weight[l_row == l_col]
        tmp_left = 1 / (self.h * l_dia + self.i)
        tmp_left.masked_fill_(tmp_left == float('inf'), 0. + 0.j)

        # Compute Jacobi matrix
        tmp_right_jacobi = self.h * l_weight.type(torch.cfloat)
        tmp_right_jacobi[l_row == l_col] = 0. + 0.j
        jacobi = -tmp_left[l_row] * tmp_right_jacobi

        # Compute b
        tmp_right_b = (l_weight * self.h).type(self.i.dtype)
        tmp_right_b[l_row == l_col] -= self.i
        b = tmp_left[l_row] * tmp_right_b
        b = torch.sparse_coo_tensor(indices=l_index, values=b, device=self.i.device)

        return jacobi, b

    def calculate_b(self, b, y_j):
        return torch.matmul(b, y_j.type(torch.cfloat))
class CayleyNet(nn.Module):
    def __init__(self, n_conv, r, K, feature_dim, hidden_dim, output_dim):
        super(CayleyNet, self).__init__()
        convs = []
        for i in range(n_conv):
            convs.append(CayleyConv(r, K, feature_dim if i == 0 else hidden_dim, hidden_dim))
            convs.append(nn.ReLU())
        self.convs = nn.ModuleList(convs)
        self.pool = TopKPooling(hidden_dim, ratio=0.9)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        for i in range(0, len(self.convs), 2):
            conv = self.convs[i]
            relu = self.convs[i + 1]
            x = conv(x, edge_index)
            x = relu(x)

        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, batch=batch)
        x = global_mean_pool(x, batch=batch)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)

        return x

def train(model, train_loader, optimizer, loss_func):
    model.train()
    for epoch in range(10):
        epoch_start_time = time.time()
        correct = 0
        total = 0
        losses = []

        for data in train_loader:
            data = data.to(DEVICE)
            out = model(data.x, data.edge_index, data.batch)
            loss = loss_func(out, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Track loss and accuracy
            losses.append(loss.item())
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

        epoch_loss = np.mean(losses)
        epoch_accuracy = correct / total
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f} | Time: {epoch_time:.2f}s")

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    N_CONV = 3
    model = CayleyNet(n_conv=N_CONV, r=3, K=3, feature_dim=7, hidden_dim=64, output_dim=dataset.num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=5)

    train(model, train_loader=train_loader, optimizer=optimizer, loss_func=criterion)
    evaluate(model, test_loader=test_loader)

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

if __name__ == '__main__':
    main()
