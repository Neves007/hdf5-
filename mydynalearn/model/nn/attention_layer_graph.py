import torch
import torch.nn as nn


class GATLayer_regular(nn.Module):

    def __init__(self,config, input_size, output_size, bias=True):
        super().__init__()
        self.config = config
        # input
        self.input_linear_layer1 = nn.Linear(input_size, output_size, bias=bias)
        self.input_linear_layer2 = nn.Linear(input_size, output_size, bias=bias)
        # attention
        self.a_1 = nn.Linear(output_size, 1, bias=bias)
        self.a_2 = nn.Linear(output_size, 1, bias=bias)
        # activation function
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.GELU()




    def forward(self, network, x0):
        """
        features : n * m dense matrix of feature vectors
        adj : n * n  sparse signed orientation matrix
        output : n * k dense matrix of new feature vectors
        """

        x0_i = self.leakyrelu(self.input_linear_layer1(x0))
        x0_j = self.leakyrelu(self.input_linear_layer2(x0))
        adj = network.inc_matrix_adj0

        indices = adj.coalesce().indices()

        a_1 = self.a_1(x0_i) # a_1：a*hi
        a_2 = self.a_2(x0_j) # a_2：a*hj

        # a_1 + a_2.T：e矩阵
        # v：e矩阵，有效e值
        attention_v = torch.sigmoid(a_1 + a_2.T)[indices[0, :], indices[1, :]]
        # e矩阵转为稀疏矩阵
        attention = torch.sparse_coo_tensor(indices, attention_v, size=adj.shape)

        # 考虑attention权重的特征。
        output = torch.sparse.mm(attention, x0_j)
        # 加上自身嵌入
        output += x0
        return output

class GraphAttentionLayer(nn.Module):
    def __init__(self,config, input_size, output_size, heads, concat, bias=True):
        super().__init__()
        self.config = config
        self.layer0_1 = torch.nn.ModuleList([GATLayer_regular(config, input_size, output_size, bias) for _ in range(heads)])



    def forward(self,**kwargs):
        x0_1 = torch.stack([gat(**kwargs) for gat in self.layer0_1])
        x0_1 = torch.mean(x0_1,dim=0)
        return x0_1