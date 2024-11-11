import yaml
import os
import random
import torch
import numpy as np
from mydynalearn.config.yaml_config.configfile import ConfigFile
from Dao import Dao

class ConfigExp():
    def __init__(self, **kwargs):
        self.IS_WEIGHT = kwargs['IS_WEIGHT']
        seed = 7
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        config_network = ConfigFile.get_config_network()
        config_dynamic = ConfigFile.get_config_dynamic()
        config_dataset = ConfigFile.get_config_dataset()
        config_optimizer = ConfigFile.get_config_optimizer()
        config_model = ConfigFile.get_config_model()
        self.network = config_network[kwargs["network"]]
        self.dynamics = config_dynamic[kwargs["dynamics"]]
        self.model = config_model[kwargs["model"]]
        self.model['model_dir'] = config_model['default'].model_dir
        self.dataset = config_dataset['default']
        self.optimizer = config_optimizer['radam']
        self.DEVICE = torch.device('cpu')
        self.NAME = "dynamicLearning-" + self.network.NAME + "-" + self.dynamics.NAME + "-" + self.model.NAME
        self.mkdir()
        self.do_fix_config(**kwargs)

    def mkdir(self):
        os.makedirs(self.model['model_dir'], exist_ok=True)

    def do_fix_config(self, *args, **kwargs):
        '''调整配置
        '''
        # network
        self.network.NUM_NODES = kwargs['NUM_NODES']
        # dynamics
        self.dynamics.SEED_FREC = kwargs['SEED_FREC']
        # dataset
        self.dataset.NUM_SAMPLES = kwargs['NUM_SAMPLES']
        self.dataset.NUM_TEST = kwargs['NUM_TEST']
        self.dataset.T_INIT = kwargs['T_INIT']
        # model
        self.model.EPOCHS = kwargs['EPOCHS']
        NUM_STATES = len(self.dynamics.STATES_MAP.keys())
        self.model.in_channels[0] = NUM_STATES
        self.model.out_channels[-1] = NUM_STATES
        self.DEVICE = kwargs['DEVICE']  # 10


