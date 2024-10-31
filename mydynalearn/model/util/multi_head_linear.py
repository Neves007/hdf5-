from torch.nn import Parameter, Linear, Sequential, Module
import torch.nn as nn
import torch
from torch_geometric.nn.inits import glorot, zeros

class MultiHeadLinear(nn.Module):
    def __init__(self, num_channels, heads=1, bias=False):
        nn.Module.__init__(self)
        self.num_channels = num_channels
        self.heads = heads

        self.weight = Parameter(torch.Tensor(heads, num_channels))  # 对应GAT论文公式(1)的a,但是只有一半。
        if bias:
            self.bias = Parameter(torch.Tensor(heads))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        x = (x * self.weight).sum(dim=-1)
        if self.bias is not None:
            return x + self.bias
        else:
            return x

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def __repr__(self):
        return "{}({}, heads={})".format(
            self.__class__.__name__, self.num_channels, self.heads
        )

