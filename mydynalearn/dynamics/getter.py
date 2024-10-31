from mydynalearn.dynamics.compartment_model import *

__DYNAMICS__ = {
    "UAU": UAU,
    "CompUAU": CoevUAU,
    "CoopUAU": CoevUAU,
    "AsymUAU": CoevUAU,
    "SCUAU": SCUAU,
    "SCCompUAU": SCCoevUAU,
    "SCCoopUAU": SCCoevUAU,
    "SCAsymUAU": SCCoevUAU,
}


def get(config):
    NAME = config.dynamics.NAME
    if NAME in __DYNAMICS__:
        return __DYNAMICS__[NAME](config)
    else:
        raise ValueError(
            f"{NAME} is invalid, possible entries are {list(__DYNAMICS__.keys())}"
        )
