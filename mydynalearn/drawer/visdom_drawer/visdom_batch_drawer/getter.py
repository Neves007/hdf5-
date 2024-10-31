from mydynalearn.drawer.visdom_drawer.visdom_batch_drawer import *

__DYNAMICS__ = {
    "UAU": VisdomBatchDrawerUAU,
    "CompUAU": VisdomBatchDrawerCompUAU,
    "SCUAU": VisdomBatchDrawerUAU,
    "SCCompUAU": VisdomBatchDrawerCompUAU,
}


def get(config,dynamics):
    NAME = config.dynamics.NAME
    if NAME in __DYNAMICS__:
        return __DYNAMICS__[NAME](dynamics)
    else:
        raise ValueError(
            f"{NAME} is invalid, possible entries are {list(__DYNAMICS__.keys())}"
        )
