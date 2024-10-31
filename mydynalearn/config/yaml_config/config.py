import yaml
import os
import random
import torch
import numpy as np
from easydict import EasyDict as edict

class Config:
    work_dir_path = 'mydynalearn/config/yaml_config/'

    config_analyze_file_name = 'config_analyze.yaml'
    config_dataset_file_name = 'config_dataset.yaml'
    config_drawer_file_name = 'config_drawer.yaml'
    config_dynamic_file_name = 'config_dynamic.yaml'
    config_model_file_name = 'config_model.yaml'
    config_network_file_name = 'config_network.yaml'
    config_optimizer_file_name = 'config_optimizer.yaml'

    @staticmethod
    def get_config_analyze():
        with open(os.path.join(Config.work_dir_path, Config.config_analyze_file_name), 'r') as file:
            config = edict(yaml.safe_load(file))
        return config

    @staticmethod
    def get_config_dataset():
        with open(os.path.join(Config.work_dir_path, Config.config_dataset_file_name), 'r') as file:
            config = edict(yaml.safe_load(file))
        return config

    @staticmethod
    def get_config_drawer():
        with open(os.path.join(Config.work_dir_path, Config.config_drawer_file_name), 'r') as file:
            config = edict(yaml.safe_load(file))
        return config

    @staticmethod
    def get_config_dynamic():
        with open(os.path.join(Config.work_dir_path, Config.config_dynamic_file_name), 'r') as file:
            config = edict(yaml.safe_load(file))
        return config

    @staticmethod
    def get_config_model():
        with open(os.path.join(Config.work_dir_path, Config.config_model_file_name), 'r') as file:
            config = edict(yaml.safe_load(file))
        return config

    @staticmethod
    def get_config_network():
        with open(os.path.join(Config.work_dir_path, Config.config_network_file_name), 'r') as file:
            config = edict(yaml.safe_load(file))
        return config

    @staticmethod
    def get_config_optimizer():
        with open(os.path.join(Config.work_dir_path, Config.config_optimizer_file_name), 'r') as file:
            config = edict(yaml.safe_load(file))
        return config