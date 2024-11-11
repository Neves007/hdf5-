import os.path
import pickle
import torch
from tqdm import *
from abc import abstractmethod
from mydynalearn.networks import *
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mydynalearn.logger import Log
from mydynalearn.util.lazy_loader import PickleLazyLoader
from Dao import DataHandler


class PeriodicDateset(DataHandler):
    '''数据集类
    通过网络network和dynamics来说生成动力学数据集

    生成数据：
        - run_dynamic_process 连续时间动力学数据
        - run 生成动力学数据
    '''

    def __init__(self, config) -> None:
        self.config = config
        self.dataset_config = config.dataset
        self.logger = Log("PeriodicDateset")
        self.init_metadata()
        parent_group = "dataset/periodic_evolution"
        cur_group = (f"NETWORK_NAME_{self.metadata['NETWORK_NAME']}_"
                     f"NUM_NODES_{self.metadata['NUM_NODES']}/"
                     f"DYNAMIC_NAME_{self.metadata['DYNAMIC_NAME']}_"
                     f"T_INIT_{self.metadata['T_INIT']}_"
                     f"SEED_FREC_{self.metadata['SEED_FREC']}"
                     )
        DataHandler.__init__(self, parent_group, cur_group)

    def get_network(self):
        network = get_network(self.config)
        return network

    def get_dynamics(self):
        dynamics = get_dynamics(self.config)
        return dynamics

    def init_metadata(self):
        metadata = {}
        metadata['DEVICE'] = self.config.DEVICE
        # network
        metadata['NETWORK_NAME'] = self.config.network['NAME']
        metadata["NUM_NODES"] = self.config.network['NUM_NODES']
        # dynamics
        metadata['DYNAMIC_NAME'] = self.config.dynamics['NAME']
        metadata["SEED_FREC"] = self.config.dynamics['SEED_FREC']
        metadata["STATES"] = list(self.config.dynamics.STATES_MAP.keys())
        # dataset
        metadata['NUM_SAMPLES'] = self.config.dataset['NUM_SAMPLES']
        metadata['T_INIT'] = self.config.dataset['T_INIT']
        metadata['IS_WEIGHT'] = self.dataset_config['IS_WEIGHT']
        self.set_metadata(metadata)

    def begin(self):
        self.NUM_SAMPLES = self.dataset_config.NUM_SAMPLES
        self.T_INIT = self.dataset_config.T_INIT
        self.DEVICE = self.config.DEVICE
        self.network = get_network(self.config)
        self.dynamics = get_dynamics(self.config)

    def _init_dataset(self):
        assert self.network.MAX_DIMENSION == self.dynamics.MAX_DIMENSION
        NUM_NODES = self.network.NUM_NODES
        NUM_STATES = self.dynamics.NUM_STATES
        NUM_SAMPLES = self.NUM_SAMPLES
        self.x0_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.y_ob_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.y_true_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.weight_T = torch.zeros(NUM_SAMPLES, NUM_NODES).to(self.config.DEVICE, dtype=torch.float)

    def _save_onesample_dataset(self, t, old_x0, new_x0, true_tp, weight, **kwargs):
        self.x0_T[t] = old_x0
        self.y_ob_T[t] = new_x0
        self.y_true_T[t] = true_tp
        self.weight_T[t] = weight

    def build_dataset(self):
        '''生成动力学数据
            简介
                - 在T_INIT时间后重置初始节点，从而增加传播动力学异质性。
        '''
        # 获取动力学数据
        self.dynamics.set_network(self.network)  # 设置动力学网络
        self._init_dataset()
        # 生成数据集
        for t in range(self.NUM_SAMPLES):
            # 动力学初始化
            if t % self.T_INIT == 0:
                self.dynamics.init_stateof_network()  # 在T_INIT时间后重置网络状态
            # 生成并存储一个样本数据集
            onestep_spread_result = self.dynamics._run_onestep()
            self.dynamics.set_features(**onestep_spread_result)
            self._save_onesample_dataset(t, **onestep_spread_result)
        dataset = {
            "x0_T": self.x0_T,
            "y_ob_T": self.y_ob_T,
            "y_true_T": self.y_true_T,
            "weight_T": self.weight_T,
        }
        self.set_dataset(dataset)
