import torch
from .radam import *


__optimizers__ = {
    "Adam": lambda config: lambda p: torch.optim.Adam(
        params=p,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
    ),
    "RAdam": lambda config: lambda p: torch.optim.RAdam(
        params=p,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    ),
}


def get(config):
    NAME = config.NAME

    if NAME in __optimizers__:
        return __optimizers__[NAME](config)
    else:
        raise ValueError(
            f"{NAME} is invalid, possible entries are {list(__optimizers__.keys())}"
        )
