import loguru
from mydynalearn.config import ConfigExp

from mydynalearn.networks import *
__networks__ = {
    "ER": ER,
    "SF": SF,
    "SCER": SCER,
    "SCSF": SCSF,
    "ToySCER": ToySCER,
    "CONFERENCE": Realnet,
    "HIGHSCHOOL": Realnet,
    "HOSPITAL": Realnet,
    "WORKPLACE": Realnet,
}


def get(config):
    if isinstance(config, ConfigExp):
        NAME = config.network.NAME
    elif isinstance(config, str):
        NAME = config
    if NAME in __networks__:
        net = __networks__[NAME](config)
        return net
    else:
        loguru.logger.error("there is no network named {}",NAME)