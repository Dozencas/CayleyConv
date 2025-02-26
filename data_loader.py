

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataListLoader
from torch.utils.data import Dataset, random_split
from pre_calc_eig import compute_and_store_eigen
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Step 1: Define a custom dataset class that incorporates eigen decomposition data
class IndexedDataset(Dataset):
    def __init__(self, root, name):
        super().__init__()
        # Load the dataset
        self.name = name
        self.dataset = TUDataset(root, name)
        
        # Load or precomputed eigen decomposition results 
        self.eig_results = self._get_eig_results()
    def _get_eig_results(self):
        eig_res_path = os.path.join('eig_results', f'{self.name}_eig.pt')
        if os.path.exists(eig_res_path):
            eig_results = load_eigen_results(eig_res_path)
        else:
            print('process and dump eig results for the first time')
            eig_results = compute_and_store_eigen(self.dataset, eig_res_path)
        return eig_results


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Fetch the data for a single graph
        data = self.dataset[idx]
        # Store original index to maintain correspondence with eigen decomposition data
        data.original_index = idx
        # Attach eigenvalues and eigenvectors to the graph data as separate fields
        data.eig_vals = self.eig_results[idx]['eig_vals']
        data.eig_vecs = self.eig_results[idx]['eig_vecs']
        return data


# Step 2: Load the eigen decomposition data from file
def load_eigen_results(file_path):
    try:
        eig_results = torch.load(file_path)
        print(f"Loaded eigen decomposition results from {file_path}")
        return eig_results
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}


def train_val_test_split(dataset, ratio, random_seed=None):
    train_set, val_set, test_set = random_split(dataset, ratio, torch.Generator().manual_seed(random_seed) if random_seed else None)
    return train_set, val_set, test_set

# Main function to initialize dataset and dataloader
def main():
    # Path to the saved eigen decomposition results
    save_path = r"./eigen_results.pt"
    eig_results = load_eigen_results(save_path)

    # Initialize the custom dataset with eigen decomposition data
    dataset = IndexedDataset(root='data/TUDataset', name='PROTEINS', eig_results=eig_results)

    # Use DataListLoader to load data in list format for each batch
    train_loader = DataListLoader(dataset, batch_size=5, shuffle=True)

    # Example iteration through the DataLoader
    for batch_data in train_loader:
        for graph_data in batch_data:
            # Access graph data including eigenvalues and eigenvectors
            print("Graph index:", graph_data.original_index)
            print("Eigenvalues:", graph_data.eig_vals)
            print("Eigenvectors shape:", graph_data.eig_vecs.shape)
            print("Eigenvectors:", graph_data.eig_vecs)
        break  # Just run one batch as an example


if __name__ == '__main__':
    main()
