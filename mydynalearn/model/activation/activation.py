import torch.nn as nn

__activations__ = {
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax(dim=-1),
    "relu": nn.GELU(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "identity": nn.Identity(),
}


def get(NAME):
    if NAME in __activations__:
        return __activations__[NAME]
    else:
        raise ValueError(
            f"{NAME} is invalid, possible entries are {list(__activations__.keys())}"
        )
