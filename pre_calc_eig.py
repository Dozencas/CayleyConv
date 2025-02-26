import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import get_laplacian
import os

def compute_and_store_eigen(dataset, save_path):
    eig_results = {}
    failed_indices = []
    success_count = 0

    for idx, data in enumerate(dataset):
        edge_weight = torch.ones(data.edge_index.size(1))
        lap_index, lap_weight = get_laplacian(data.edge_index, edge_weight, normalization=None, num_nodes=data.num_nodes)
        N = data.num_nodes
        k = min(max(int(N * 0.2), 6), N - 1)
        sparse_lap = coo_matrix((lap_weight.cpu(), lap_index.cpu()), shape=(N, N))

        try:
            eig_vals, eig_vecs = eigsh(sparse_lap, k=k, which='SM')
            eig_vals = torch.from_numpy(eig_vals)
            eig_vecs = torch.from_numpy(eig_vecs)
            # record the original idx
            eig_results[idx] = {'eig_vals': eig_vals, 'eig_vecs': eig_vecs, 'original_index': idx}
            success_count += 1
        except Exception as e:
            print(f"Error in eigendecomposition for graph index {idx}: {e}")
            failed_indices.append(idx)

    # 打印成功和失败的索引
    print(f"Successfully processed eigen decomposition for {success_count} graphs.")
    if failed_indices:
        print(f"Failed to compute eigen decomposition for graph indices: {failed_indices}")

    torch.save(eig_results, save_path)
    print(f"Eigen decomposition results saved to {save_path}")
    return eig_results

if __name__ == '__main__':
    save_path = r'./eigen_results_mutag.pt'
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    compute_and_store_eigen(dataset, save_path)
