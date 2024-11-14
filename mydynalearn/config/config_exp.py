import yaml
import os
import random
import torch
import numpy as np
from mydynalearn.config.yaml_config.configfile import ConfigFile
from Dao import Dao


class ConfigExp():
    def __init__(self,
                 NETWORK_NAME,
                 DYNAMIC_NAME,
                 MODEL_NAME,
                 IS_WEIGHT=False,
                 DEVICE='cuda', **kwargs):
        self.IS_WEIGHT = IS_WEIGHT
        seed = 7
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        config_network = ConfigFile.get_config_network()
        config_dynamic = ConfigFile.get_config_dynamic()
        config_dataset = ConfigFile.get_config_dataset()
        config_optimizer = ConfigFile.get_config_optimizer()
        config_model = ConfigFile.get_config_model()
        self.network = config_network[NETWORK_NAME]
        self.dynamics = config_dynamic[DYNAMIC_NAME]
        self.model = config_model[MODEL_NAME]
        self.model['model_dir'] = config_model['default'].model_dir
        self.dataset = config_dataset['default']
        self.optimizer = config_optimizer['radam']
        self.DEVICE = DEVICE
        self.mkdir()
        self.do_fix_config(**kwargs)

    def mkdir(self):
        os.makedirs(self.model['model_dir'], exist_ok=True)

    def do_fix_config(self, **kwargs):
        '''调整配置
        '''
        # network
        self.network.NUM_NODES = kwargs.get('NUM_NODES', self.network.NUM_NODES)

        # dynamics
        self.dynamics.SEED_FREC = kwargs.get('SEED_FREC', self.dynamics.SEED_FREC)

        # dataset
        self.dataset.NUM_SAMPLES = kwargs.get('NUM_SAMPLES', self.dataset.NUM_SAMPLES)
        self.dataset.NUM_TEST = kwargs.get('NUM_TEST', self.dataset.NUM_TEST)
        self.dataset.T_INIT = kwargs.get('T_INIT', self.dataset.T_INIT)

        # model
        self.model.EPOCHS = kwargs.get('EPOCHS', self.model.EPOCHS)

        # 计算 NUM_STATES 并更新 in_channels 和 out_channels
        NUM_STATES = len(self.dynamics.STATES_MAP.keys())
        self.model.in_channels[0] = NUM_STATES
        self.model.out_channels[-1] = NUM_STATES
        self.NAME = "_".join(
            [
                "Config",
                "network=" + self.network.NAME,
                "dynamics=" + self.dynamics.NAME,
                "model=" + self.model.NAME,
                "SEEDFREC=" + str(self.dynamics.SEED_FREC),
                "NUMSAMPLES=" + str(self.dataset.NUM_SAMPLES),
                "TINIT=" + str(self.dataset.T_INIT),
            ])
