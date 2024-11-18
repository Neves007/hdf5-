from mydynalearn.model.nn.nnlayers import *
import torch.nn as nn
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics

class GraphAttentionModel(nn.Module):
    def __init__(self, config):
        """Dense version of GAT."""
        super(GraphAttentionModel,self).__init__()
        self.config = config
        self._nnlayer = nnLayer(config)
        self.in_layer = self._nnlayer.get_in_layers()
        self.gat_layer_1 = self._nnlayer.get_gnn_layer()
        self.out_layers = self._nnlayer.get_out_layers()
        self.DEVICE = config.DEVICE

    def set_netowrk(self):
        self.network = get_network(self.config)
        self.network.load()
    def set_dynamics(self):
        self.dynamics = get_dynamics(self.config)

    def forward(self,x0, y_ob, y_true, weight):
        if not hasattr(self, "network"):
            self.set_netowrk()
        # 数据预处理
        x0 = x0.squeeze()
        y_ob = y_ob.squeeze()
        y_true = y_true.squeeze()
        weight = weight.squeeze()
        # attention
        x0_in = self.in_layer(x0)
        args = {
            "network": self.network,
            "x0": x0_in}
        x = self.gat_layer_1(**args)
        out = self.out_layers(x)
        return x0,out,y_true,y_ob, weight

