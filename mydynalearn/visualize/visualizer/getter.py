from . import *
__visulizer__ = {
    "FigYtrureYpred": FigYtrureYpredDrawer,
    "LossK1K2": LossK1K2Drawer
}



def get(visulizer_name):
    assert visulizer_name in __visulizer__, f"Error: {visulizer_name} is not in __visulizer__"
    return __visulizer__[visulizer_name]

