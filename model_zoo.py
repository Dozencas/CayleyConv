
#######代码不收敛，因为拉普拉斯矩阵有问题 接近0修改代码


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from torch.nn import Parameter
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix

from torch_geometric.nn import MessagePassing, TopKPooling, global_mean_pool
from torch_geometric.utils import degree, get_laplacian



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
        self.h = Parameter(torch.tensor(0.1, dtype=torch.float))
        self.i = Parameter(torch.tensor(0. + 1.j), requires_grad=False)
        self.alpha = Parameter(torch.tensor(0.1, dtype=torch.float))
        #C_0
        # self.real_linear = nn.Linear(in_channels, out_channels, bias=False)
        # #C_1...C_r
        # self.complex_linears = nn.ModuleList([ComplexLinear(in_channels, out_channels, bias=False) for _ in range(r)])
        self.c_0 = Parameter(torch.tensor(0.1, dtype=torch.float))
        self.c_j = Parameter(torch.tensor(torch.full([self.r], torch.tensor(0.1 + 0.1j)), dtype=torch.cfloat))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # stdv = 1. / math.sqrt(self.in_channels)
        # self.h.data = 1.0
        # self.alpha.data = 0.0

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
    ) -> Tensor:

        y_j = x
        # out = self.real_linear(y_j)
        out = self.c_0 * y_j


        n_nodes = x.size(self.node_dim)
        device = edge_index.device
        edge_weight = torch.ones(edge_index.size(1), device=device)
        # Laplacian
        l_index, l_weight = get_laplacian(edge_index, edge_weight, normalization=None, num_nodes=n_nodes)
        l_row, l_col = l_index
        dia_index = l_row == l_col
        l_weight[dia_index] -= self.alpha

        # Jacobi
        jacobi, b = self.calculate_jacobi_and_b(l_index, l_weight, dia_index)


        # calcualte r polynomials
        for j in range(self.r):

            b_j = self.calculate_b(b, y_j)
            y_j_k = b_j

            # K jacobi iteration
            for k in range(self.K):
                # y_j ^ k+1 = J @ y_j ^ k + b_j
                y_j_k = self.propagate(l_index, x=y_j_k, jacobi=jacobi) + b_j
            y_j = y_j_k
            out = out + 2 * (self.c_j[j] * y_j).real
        return out

    def message(self, x_j: Tensor, jacobi: Tensor) -> Tensor:
        # J Y
        return jacobi.view(-1, 1) * x_j

    def calculate_jacobi_and_b(self, l_index, l_weight, dia_index):
        l_row, l_col = l_index

        # 计算 (hD + iI)^-1
        l_dia = l_weight[dia_index]
        tmp_left = 1 / (self.h * l_dia + self.i)
        tmp_left.masked_fill_(tmp_left == float('inf'), 0. + 0.j)

        # 计算 Jacobi 矩阵
        tmp_right_jacobi = self.h * l_weight.type(torch.cfloat)
        tmp_right_jacobi[dia_index] = 0. + 0.j
        jacobi = -tmp_left[l_row] * tmp_right_jacobi

        # 计算 b
        tmp_right_b = (l_weight * self.h).type(self.i.dtype)
        tmp_right_b[dia_index] -= self.i
        b = tmp_left[l_row] * tmp_right_b
        b = torch.sparse_coo_tensor(indices=l_index, values=b, device=self.i.device)

        return jacobi, b

    def calculate_b(self, b, y_j):
        return torch.matmul(b, y_j.type(torch.cfloat))



class CayleyConvLanczos(nn.Module):
    def __init__(
            self,
            r: int,
            in_channels: int,
            out_channels: int,
            eig_ratio: float = 0.2,
            **kwargs,
    ):
        super().__init__(**kwargs)

        assert r > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.r = r
        self.eig_ratio = eig_ratio
        self.h = Parameter(torch.tensor(0.1, dtype=torch.float))
        self.i = Parameter(torch.tensor(0. + 1.j), requires_grad=False)
        self.alpha = Parameter(torch.tensor(0.1, dtype=torch.float))
        #C_0
        # self.c_0 = Parameter(torch.randn((out_channels, in_channels)))
        #C_1...C_r
        # self.c_j = Parameter(torch.randn((r, out_channels, in_channels), dtype=torch.cfloat))
        self.c_0 = Parameter(torch.tensor(0.1, dtype=torch.float))
        self.c_j = Parameter(torch.tensor(torch.full([self.r], torch.tensor(0.1 + 0.1j)), dtype=torch.cfloat))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        #print("x shape:", x.shape)  # 打印输入特征的形状
        in_channels = x.shape[-1]
        #print("in_channels:", in_channels)  # 打印输入特征的维度
        #print("out_channels:", self.out_channels)  # 打印输出特征的维度
        n_nodes = x.size(0)  # m
        k = max(int(n_nodes * self.eig_ratio), 6)  # number of eigen vals in lanczos method

        device = edge_index.device
        edge_weight = torch.ones(edge_index.size(1), device=device)
        # Laplacian
        l_index, l_weight = get_laplacian(edge_index, edge_weight, normalization=None, num_nodes=n_nodes)
        eig_vals, eig_vecs = self.eigen_decomposition(l_index, l_weight, n_nodes, k, x.device)  # Ensure same device

        res = torch.zeros((x.shape[0], self.out_channels), dtype=torch.float, device=x.device)  # (m * d_out)

        for i in range(k):
            eigen_filter = self.get_eigen_cayley_filter(eig_vals[i], x.device)  # Ensure eigen filter is on same device
            #print("eigen_filter shape:", eigen_filter.shape)  # 打印

            v_i = eig_vecs[:, i].reshape(-1, 1).to(x.device)  # Ensure v_i is on the same device as x
            #print("eig_vecs shape:", eig_vecs.shape)

            # 确保矩阵乘法形状正确
            vi_t_x = (v_i.T @ x)  # (1, 160)
            #("vi_t_x shape:", vi_t_x.shape)  # 应该是 (1, 7)

            # 进行矩阵乘法
            # vi_t_x_Q = vi_t_x @ eigen_filter.T  # (1, 7) @ (7, 64) => (1, 64)
            vi_t_x_Q = vi_t_x * eigen_filter
            #print("vi_t_x_Q shape:", vi_t_x_Q.shape)  # 应该是 (1, 64)

            res += v_i @ vi_t_x_Q

        return res

    def get_eigen_cayley_filter(self, eig_val, device):
        """
        Returns the Cayley filter based on eigenvalue and device.
        """
        base = (self.h * (eig_val - self.alpha) - self.i) / (self.h * (eig_val - self.alpha) + self.i)
        r_complex = torch.stack([self.c_j[j - 1] * base ** j for j in range(1, 1 + self.r)]).sum(0)  # (r, )
        return self.c_0 + 2 * r_complex.real

    def eigen_decomposition(self, lap_index, lap_weight, n_nodes, k, device):
        lap_index = lap_index.cpu().numpy()  # 转移到 CPU
        lap_weight = lap_weight.cpu().numpy()  # 转移到 CPU

        # 添加正则化项
        # epsilon = 1e-6  # 正则化常数
        # lap_weight += epsilon  # 在所有权重上加上一个小常数

        sparse_lap = coo_matrix((lap_weight, lap_index), shape=(n_nodes, n_nodes))

        try:
            vals, vecs = eigsh(sparse_lap, k=k, which='SM', maxiter=50000)

        except Exception as e:
            print(f"特征值分解失败: {e}")
            raise

        return torch.from_numpy(vals).to(device), torch.from_numpy(vecs).to(device)


class CayleyNet(nn.Module):
    def __init__(self, n_conv, r, K, feature_dim, hidden_dim, output_dim, conv_type='Jacobian'):
        super(CayleyNet, self).__init__()
        convs = []
        for i in range(n_conv):
            # convs.append(CayleyConv(r, K, feature_dim if i == 0 else hidden_dim, hidden_dim))
            conv = CayleyConv(r, K, feature_dim if i == 0 else hidden_dim, hidden_dim) if conv_type == 'Jacobian' else CayleyConvLanczos(r, feature_dim if i == 0 else hidden_dim, hidden_dim)

            convs.append(conv)
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

