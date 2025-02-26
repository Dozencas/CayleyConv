import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
from model_zoo import CayleyConv, CayleyConvLanczos


def evaluate_forward(conv1, conv2, test_loader, device='cpu'):
    conv1.eval()  # 设置模型为评估模式
    conv2.eval()

    l2_diffs = []  # 用来保存每次的 L2 差异
    cosine_sims = []  # 用来保存每次的余弦相似度

    with torch.no_grad():  # 禁用梯度计算，节省内存
        for data in test_loader:
            data = data.to(device)  # 确保数据在正确的设备上
            out_1 = conv1(data.x, data.edge_index)  # 进行第一个模型的前向传播
            out_2 = conv2(data.x, data.edge_index)  # 进行第二个模型的前向传播

            # 计算 L2 距离
            l2_diff = torch.norm(out_1 - out_2)
            l2_diffs.append(l2_diff.item())

            # 计算余弦相似度
            cos_sim = F.cosine_similarity(out_1, out_2, dim=-1).mean()
            cosine_sims.append(cos_sim.item())


    avg_l2_diff = sum(l2_diffs) / len(l2_diffs) if l2_diffs else 0
    avg_cosine_sim = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0

    print("Average L2 Difference between the two models' outputs: {:.4f}".format(avg_l2_diff))  # 输出平均 L2 差异
    print("Average Cosine Similarity between the two models' outputs: {:.4f}".format(avg_cosine_sim))  # 输出平均余弦相似度

def main():
    # 假设 `DEVICE` 已经被定义为 'cuda' 或 'cpu'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载数据集
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    test_dataset = dataset

    # 确保模型初始化时特征维度正确
    conv1 = CayleyConv(2, 10, dataset.num_node_features, dataset.num_node_features).to(DEVICE)
    conv2 = CayleyConvLanczos(2, dataset.num_node_features, dataset.num_node_features).to(DEVICE)

    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 运行评估过程，比较两个模型的输出
    evaluate_forward(conv1, conv2, test_loader, device=DEVICE)


if __name__ == '__main__':
    main()
