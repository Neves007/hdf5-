import loguru

from mydynalearn.model.nn import *
__gnn__ = {
    "GAT": GraphAttentionLayer,
    "SAT": SimplexAttentionLayer,
    "DiffSAT": SimplexDiffAttentionLayer,
    "DualSAT": SimplexDualAttentionLayer,
}


def get(config):
    NAME = config.model.NAME
    if NAME in __gnn__:
        gnn = __gnn__[NAME]
        return gnn
    else:
        loguru.logger.error("there is no Gnn model named {}",NAME)