
import torch
from data_loader import IndexedDataset, train_val_test_split  # 调用已定义的 IndexedDataset 和 load_eigen_results
from model import CayleyNet
import torch.optim as optim
from torch_geometric.loader.data_list_loader import DataListLoader

from tqdm import tqdm
from copy import deepcopy
import numpy as np
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_and_evaluate(model: CayleyNet, train_loader, val_loader, test_loader, optimizer, criterion):
    model.train()
    epoch_losses = []
    max_acc = -float('inf')
    best_model = None
    best_epoch = 0

    for epoch in range(10):
        losses = []
        for batch_data in tqdm(train_loader):
            batch_data = [data.to(DEVICE) for data in batch_data]
            optimizer.zero_grad()

            # Forward pass and loss computation
            for data in batch_data:
                out = model(data)
                # y = torch.cat([data.y for data in batch_data], dim=0).to(DEVICE)
                y = data.y
                loss = criterion(out, y) / len(batch_data)

                # Backward pass to compute gradients for this batch
                loss.backward()
                losses.append(loss.cpu().item())

            optimizer.step()

        avg_epoch_loss = np.mean(losses)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}, Train Avg Loss: {avg_epoch_loss:.4f}")
        val_acc, val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}, Val Avg Loss: {val_loss:.4f}; Val Acc: {val_acc:.4f}")
        if val_acc > max_acc:
            max_acc = val_acc
            best_model = deepcopy(model.state_dict())
            best_epoch = epoch + 1

    model.load_state_dict(best_model)
    test_acc, _ = evaluate(model, test_loader, criterion)
    print(f"Best epoch: {best_epoch}; Test Accuracy: {test_acc:.4f}")
    



def evaluate(model: CayleyNet, loader, criterion):
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data in loader:
            batch_data = [data.to(DEVICE) for data in batch_data]
            for data in batch_data:
                out = model(data)
                # y = torch.cat([data.y for data in batch_data], dim=0).to(DEVICE)
                y = data.y
                loss = criterion(out, y)
                losses.append(loss.cpu().item())
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += 1
    accuracy = correct / total if total > 0 else 0
    avg_loss = np.mean(losses)
    return accuracy, avg_loss



def main():
    dataset = IndexedDataset(root='data/TUDataset', name='MUTAG')
    train_set, val_set, test_set = train_val_test_split(dataset, ratio=[0.7, 0.15, 0.15], random_seed=1)
    train_loader = DataListLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataListLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataListLoader(test_set, batch_size=1, shuffle=False)


    model = CayleyNet(
        n_conv=3,
        r=3,
        feature_dim=dataset.dataset.num_features, 
        hidden_dim=64, 
        output_dim=dataset.dataset.num_classes, 
        conv_type="jacobi"
    ).to(DEVICE)


    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()


    train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, criterion)


if __name__ == '__main__':
    main()


