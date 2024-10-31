from mydynalearn.drawer.visdom_drawer.visdom_epoch_drawer import *

__DYNAMICS__ = {
    "UAU": VisdomEpochDrawer,
    "CompUAU": VisdomEpochDrawer,
    "SCUAU": VisdomEpochDrawer,
    "SCCompUAU": VisdomEpochDrawer,
}


def get(config):
    NAME = config.dynamics.NAME
    if NAME in __DYNAMICS__:
        return __DYNAMICS__[NAME]()
    else:
        raise ValueError(
            f"{NAME} is invalid, possible entries are {list(__DYNAMICS__.keys())}"
        )