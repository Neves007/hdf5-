import yaml
import os
import random
import torch
import numpy as np
from mydynalearn.config.yaml_config.configfile import ConfigFile
from Dao import Dao

class ConfigTrainingExp:
    def __init__(self,
                 NAME,
                 network,
                 dynamics,
                 root_dir,
                 IS_WEIGHT=False,
                 seed=None,
                 ):
        self.NAME = NAME
        self.IS_WEIGHT = IS_WEIGHT
        self.seed = seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        config_network = ConfigFile.get_config_network()
        config_dynamic = ConfigFile.get_config_dynamic()
        config_dataset = ConfigFile.get_config_dataset()
        config_optimizer = ConfigFile.get_config_optimizer()
        config_model = ConfigFile.get_config_model()


        self.network = config_network[network]
        self.dynamics = config_dynamic[dynamics]
        self.dataset = config_dataset['default']
        self.optimizer = config_optimizer['radam']
        self.model = config_model['default']
        self.DEVICE = torch.device('cpu')

    def set_path(self, root_dir="./output"):
        with Dao() as dao:
            print("build dao.")