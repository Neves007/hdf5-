from mydynalearn.model.util.inout_linear_layers import inLinearLayers,outLinearLayers
from mydynalearn.model.nn import GraphAttentionLayer,SimplexAttentionLayer
from mydynalearn.model.nn.getter import get as gnn_getter


class nnLayer():
    def __init__(self,config):
        self.config = config
        self.DEVICE = config.DEVICE
        # in/out layer config
        self._in_channels = config.model.in_channels
        self._out_channels = config.model.out_channels
        # gnn layer config
        self._gnn_in_features = config.model.gnn_channels
        self._gnn_out_features = config.model.gnn_channels
        self._gnn_heads = config.model.heads
        self._concat = config.model.concat



    def get_in_layers(self):
        in_layer = inLinearLayers(self._in_channels).to(self.DEVICE)
        return in_layer

    def get_out_layers(self):
        out_layer = outLinearLayers(self._out_channels).to(self.DEVICE)
        return out_layer

    def get_gnn_layer(self):
        GNNLayers = gnn_getter(self.config)
        gnn_layers = GNNLayers(self._gnn_in_features,
                               self._gnn_out_features,
                               self._gnn_heads,
                               self._concat).to(self.DEVICE)
        return gnn_layers

