from mydynalearn.model.nn.nnlayers import *
import torch.nn as nn
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics

class SimplicialAttentionModel(nn.Module):
    def __init__(self, config):
        """Dense version of GAT."""
        super(SimplicialAttentionModel,self).__init__()
        self.config = config
        self._nnlayer = nnLayer(config)
        self.in_layer0 = self._nnlayer.get_in_layers()
        self.in_layer1 = self._nnlayer.get_in_layers()
        self.in_layer2 = self._nnlayer.get_in_layers()
        self.sat_layers = self._nnlayer.get_gnn_layer()
        self.out_layers = self._nnlayer.get_out_layers()
        self.DEVICE = config.DEVICE
    def set_netowrk(self):
        self.network = get_network(self.config)
        self.network.load()
    def set_dynamics(self):
        self.dynamics = get_dynamics(self.config)

    def forward(self, x0, y_ob, y_true, weight):
        if not hasattr(self, "network"):
            self.set_netowrk()
        if not hasattr(self, "dynamics"):
            self.set_dynamics()
        # 数据预处理
        x0 = x0.squeeze().to(self.DEVICE)
        y_ob = y_ob.squeeze().to(self.DEVICE)
        y_true = y_true.squeeze().to(self.DEVICE)
        weight = weight.squeeze().to(self.DEVICE)
        # 高阶信息
        x1 =self.dynamics.get_x1_from_x0(x0, self.network)
        # 只考虑edge_index
        x0_in = self.in_layer0(x0)
        x1_in = self.in_layer1(x1)

        if self.network.MAX_DIMENSION ==2:
            x2 =self.dynamics.get_x2_from_x0(x0, self.network)
            x2_in = self.in_layer2(x2)
            sat_args = {
                "network":self.network,
                "x0":x0_in,
                "x1":x1_in,
                "x2":x2_in}
        else:
            sat_args = {
                "network": self.network,
                "x0": x0_in,
                "x1": x1_in}
        x = self.sat_layers(**sat_args)

        out = self.out_layers(x)
        return x0,out,y_true,y_ob, weight
