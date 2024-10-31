


import torch

import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from mydynalearn.model.util.multi_head_linear import MultiHeadLinear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

# 重新命名
class SATLayer_regular(nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
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

        self.LinearAgg = nn.Linear(2*output_size, output_size, bias=bias)
        # output
        self.output_linear_layer1 = nn.Linear(output_size, output_size, bias=bias)
        # activation function
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.GELU()

        # output attention
        self.wq = nn.Linear(output_size, 1, bias=bias)
        self.wk = nn.Linear(output_size, 1, bias=bias)
        self.wv = nn.Linear(output_size, output_size, bias=bias)
    def attention_agg(self, xj, a_i,a_j,inc_matrix_adj):
        '''图注意力机制

        :param xj: 节点特征
        :param a_i: 节点注意力
        :param a_j: 节点i的邻居【节点，边，三角】注意力
        :param inc_matrix_adj: 节点与邻居的关联矩阵
        :return:
        '''
        indices = inc_matrix_adj.coalesce().indices()

        # a_1 + a_2.T：e矩阵
        # v：e矩阵，有效e值
        attention_v = torch.sigmoid(a_i + a_j.T)[indices[0, :], indices[1, :]]
        # e矩阵转为稀疏矩阵
        attention = torch.sparse_coo_tensor(indices, attention_v,size=inc_matrix_adj.shape)

        # 考虑attention权重的特征。
        output = torch.sparse.mm(attention, xj)
        return output

    def dual_att_high(self,x0,agg0,agg1,agg2):
        q_x0 = self.wq(x0)


        k_x0 = self.wk(x0)
        k_agg0 = self.wk(agg0)
        k_agg1 = self.wk(agg1)
        k_agg2 = self.wk(agg2)


        v_x0 = self.wv(x0)
        v_agg0 = self.wv(agg0)
        v_agg1 = self.wv(agg1)
        v_agg2 = self.wv(agg2)

        alpha_x0_x0 = torch.sigmoid(q_x0 * k_x0)
        alpha_x0_agg0 = torch.sigmoid(q_x0 * k_agg0)
        alpha_x0_agg1 = torch.sigmoid(q_x0 * k_agg1)
        alpha_x0_agg2 = torch.sigmoid(q_x0 * k_agg2)

        output = (alpha_x0_x0 * v_x0 + alpha_x0_agg0 * v_agg0 + alpha_x0_agg1 * v_agg1 + alpha_x0_agg2 * v_agg2)/4

        return output

    def forward(self,network, x0, x1, x2=None):
        """
        features : n * m dense matrix of feature vectors
        adj : n * n  sparse signed orientation matrix
        output : n * k dense matrix of new feature vectors
        """
        if x2==None:
            incMatrix_adj0, incMatrix_adj1 = network._unpack_inc_matrix_adj_info()
            xi_0 = self.leakyrelu(self.input_linear_layer1(x0))
            xj_0 = self.leakyrelu(self.input_linear_layer2(x0))

            ai_0 = self.a_1(xi_0)  # a_1：a*hi
            aj_0 = self.a_2(xj_0)  # a_2：a*hj

            agg0 = self.attention_agg(xj_0, ai_0, aj_0, incMatrix_adj0)

            output = self.layer_norm1(agg0 + x0)
        else:
            incMatrix_adj0, incMatrix_adj1, incMatrix_adj2 = network._unpack_inc_matrix_adj_info()
            xi_0 = self.leakyrelu(self.input_linear_layer1(x0))
            xj_0 = self.leakyrelu(self.input_linear_layer2(x0))
            xj_1 = self.leakyrelu(self.input_linear_layer3(x1))
            xj_2 = self.leakyrelu(self.input_linear_layer4(x2))

            ai_0 = self.a_1(xi_0)  # a_1：a*hi
            aj_0 = self.a_2(xj_0)  # a_2：a*hj
            aj_1 = self.a_3(xj_1)  # a_2：a*hj
            aj_2 = self.a_4(xj_2)  # a_2：a*hj
            agg0 = self.attention_agg(xj_0, ai_0, aj_0, incMatrix_adj0)
            agg1 = self.attention_agg(xj_1, ai_0, aj_1, incMatrix_adj1)
            agg2 = self.attention_agg(xj_2, ai_0, aj_2, incMatrix_adj2)

            output = self.layer_norm1(self.dual_att_high(x0,agg0,agg1,agg2) + x0)
        output = self.layer_norm2(self.output_linear_layer1(output)+output)
        return output

class SimplexDualAttentionLayer(nn.Module):
    def __init__(self, input_size, output_size, heads, concat, bias=True):
        super().__init__()
        self.layer0_1 = torch.nn.ModuleList([SATLayer_regular(input_size, output_size, bias) for _ in range(heads)])


    def forward(self, **kwargs):
        x0_1 = torch.stack([sat(**kwargs) for sat in self.layer0_1])
        x0_1 = torch.mean(x0_1, dim=0)
        return x0_1