import loguru

from mydynalearn.model.attention_model import GraphAttentionModel,SimplicialAttentionModel
__Model__ = {
    "GAT": GraphAttentionModel,
    "SAT": SimplicialAttentionModel,
    "DiffSAT": SimplicialAttentionModel,
    "DualSAT": SimplicialAttentionModel,
}


def get(config):
    NAME = config.model.NAME
    if NAME in __Model__:
        model = __Model__[NAME](config)
        return model
    else:
        loguru.logger.error("there is no model named {}",NAME)