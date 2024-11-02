import yaml
import os
import random
import torch
import numpy as np
from mydynalearn.config.yaml_config.configfile import ConfigFile
from Dao import Dao

class ConfigExp():
    def __init__(self,
                 network,
                 dynamics,
                 model,
                 IS_WEIGHT=False,
                 seed=2,
                 ):
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
        self.model = config_model[model]
        self.model['model_dir'] = config_model['default'].model_dir
        self.dataset = config_dataset['default']
        self.optimizer = config_optimizer['radam']
        self.DEVICE = torch.device('cpu')
        self.NAME = "dynamicLearning-" + self.network.NAME + "-" + self.dynamics.NAME + "-" + self.model.NAME
        self.mkdir()

    def mkdir(self):
        os.makedirs(self.model['model_dir'], exist_ok=True)

    def do_fix_config(self, fix_config):
        '''调整配置
        '''
        # T总时间步
        self.dataset.NUM_SAMPLES = fix_config['NUM_SAMPLES']
        self.dataset.NUM_TEST = fix_config['TESTSET_TIMESTEP']
        self.model.EPOCHS = fix_config['EPOCHS']  # 10
        NUM_STATES = len(self.dynamics.STATES_MAP.keys())
        self.model.in_channels[0] = NUM_STATES
        self.model.out_channels[-1] = NUM_STATES
        self.DEVICE = fix_config['DEVICE']  # 10

