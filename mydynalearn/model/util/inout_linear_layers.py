import torch.nn

from mydynalearn.model.nn.attention_layer_graph import *
from torch.nn import Sequential


class inLinearLayers(nn.Module):
    def __init__(self,in_channels):
        super(inLinearLayers, self).__init__()
        self.in_channels = in_channels
        self.activation = nn.GELU()
        self.template = lambda fin, fout: nn.Linear(fin, fout, bias=True)
        self.dropout_layer = torch.nn.Dropout(0.5)
        self.build()

    def forward(self, x):
        x = self.layers(x)
        return x
    def build(self):
        layers = []
        for f in zip(self.in_channels[:-1], self.in_channels[1:]):
            layers.append(self.template(*f))
            # layers.append(self.dropout_layer)
            layers.append(self.activation)
        self.layers = Sequential(*layers)

class outLinearLayers(nn.Module):
    def __init__(self,out_channels):
        super(outLinearLayers, self).__init__()
        self.out_channels = out_channels
        self.activation = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)
        self.template = lambda fin, fout: nn.Linear(fin, fout, bias=True)
        self.build()

    def forward(self, x):
        x = self.layers(x)
        return x
    def build(self):
        layers = []
        for f in zip(self.out_channels[:-2], self.out_channels[1:-1]):
            layers.append(self.template(*f))
            layers.append(self.activation)
        for f in zip(self.out_channels[-2:], self.out_channels[-1:]):
            layers.append(self.template(*f))
            layers.append(self.softmax)
        self.layers = Sequential(*layers)