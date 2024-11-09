import loguru
from mydynalearn.config import ConfigExp
from mydynalearn.model.model_attention import GraphAttentionModel,SimplicialAttentionModel
__Model__ = {
    "GAT": GraphAttentionModel,
    "SAT": SimplicialAttentionModel,
}


def get(config):
    if isinstance(config, ConfigExp):
        NAME = config.model.NAME
    elif isinstance(config, str):
        NAME = config
    if NAME in __Model__:
        model = __Model__[NAME](config)
        return model
    else:
        loguru.logger.error("there is no model named {}",NAME)