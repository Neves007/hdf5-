


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from mydynalearn.model.util.multi_head_linear import MultiHeadLinear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

class SATLayer_regular(nn.Module):

    def __init__(self,config, input_size, output_size, bias=True):
        super().__init__()
        self.config = config
        # input
        self.input_linear_layer1 = nn.Linear(input_size, output_size, bias=bias)
        self.input_linear_layer2 = nn.Linear(input_size, output_size, bias=bias)
        self.input_linear_layer3 = nn.Linear(input_size, output_size, bias=bias)
        self.input_linear_layer4 = nn.Linear(input_size, output_size, bias=bias)
        # attention
        self.a_1 = nn.Linear(output_size, 1, bias=bias)
        self.a_2 = nn.Linear(output_size, 1, bias=bias)
        self.a_3 = nn.Linear(output_size, 1, bias=bias)
        self.a_4 = nn.Linear(output_size, 1, bias=bias)
        
        self.layer_norm1 = nn.LayerNorm(output_size)
        self.layer_norm2 = nn.LayerNorm(output_size)

        self.agg_weight = nn.Parameter(torch.randn(3))
        self.LinearAgg = nn.Linear(2*output_size, output_size, bias=bias)
        # output
        self.output_linear_layer1 = nn.Linear(output_size, output_size, bias=bias)
        # activation function
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.GELU()

    def attention_agg(self, xj, a_i,a_j,inc_matrix_adj):
        indices = inc_matrix_adj.coalesce().indices()
        # a_1 + a_2.T：e矩阵
        # v：e矩阵，有效e值
        attention_v = torch.sigmoid(a_i + a_j.T)[indices[0, :], indices[1, :]]
        # e矩阵转为稀疏矩阵
        attention = torch.sparse_coo_tensor(indices, attention_v,size=inc_matrix_adj.shape)
        # 考虑attention权重的特征。
        output = torch.sigmoid(torch.sparse.mm(attention, xj))
        return output

    def forward(self,network, x0, x1, x2=None):
        """
        features : n * m dense matrix of feature vectors
        adj : n * n  sparse signed orientation matrix
        output : n * k dense matrix of new feature vectors
        """
        if x2==None:
            incMatrix_adj0 = network.inc_matrix_adj0
            xi_0 = self.leakyrelu(self.input_linear_layer1(x0))
            xj_0 = self.leakyrelu(self.input_linear_layer2(x0))


            ai_0 = self.a_1(xi_0)  # a_1：a*hi
            aj_0 = self.a_2(xj_0)  # a_2：a*hj

            agg0 = self.attention_agg(xj_0, ai_0, aj_0, incMatrix_adj0)

            # residual + layer norm
            output = self.output_linear_layer1(agg0) + xi_0
        else:
            incMatrix_adj0 = network.inc_matrix_adj0
            incMatrix_adj1 = network.inc_matrix_adj1
            incMatrix_adj2 = network.inc_matrix_adj2
            xi_0 = self.leakyrelu(self.input_linear_layer1(x0))
            xj_0 = self.leakyrelu(self.input_linear_layer2(x0))
            xj_1 = self.leakyrelu(self.input_linear_layer3(x1))
            xj_2 = self.leakyrelu(self.input_linear_layer4(x2))

            ai_0 = self.a_1(xi_0)  # a_1：a*hi
            aj_0 = self.a_2(xj_0)  # a_2：a*hj
            aj_1 = self.a_3(xj_1)  # a_2：a*hj
            aj_2 = self.a_4(xj_2)  # a_2：a*hj
            agg0 = self.attention_agg(xj_0, ai_0, aj_0, incMatrix_adj0)
            # agg1 = self.attention_agg(xj_1, ai_0, aj_1, incMatrix_adj1)
            agg2 = self.attention_agg(xj_2, ai_0, aj_2, incMatrix_adj2)

            # residual
            output = self.output_linear_layer1(agg0 + agg2)
        # add & norm
        output = self.layer_norm1(output + x0)
        return output

class SimplexAttentionLayer(nn.Module):
    def __init__(self, config, input_size, output_size, heads, concat, bias=True):
        super().__init__()
        self.config = config
        self.layer0_1 = torch.nn.ModuleList([SATLayer_regular(config, input_size, output_size, bias) for _ in range(heads)])


    def forward(self, **kwargs):
        x0_1 = torch.stack([sat(**kwargs) for sat in self.layer0_1])
        x0_1 = torch.mean(x0_1, dim=0)
        return x0_1