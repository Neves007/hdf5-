import torch
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics
from mydynalearn.logger import Log
from Dao import DataHandler
from mydynalearn.dataset.periodic_dateset import PeriodicDateset

class ContinuousDataset(PeriodicDateset):
    '''数据集类
    通过网络network和dynamics来说生成动力学数据集

    生成数据：
        - run_dynamic_process 连续时间动力学数据
        - run 生成动力学数据
    '''
    def __init__(self, config) -> None:
        PeriodicDateset.__init__(self, config)
        self.config = config
        self.dataset_config = config.dataset
        self.logger = Log("ContinuousDataset")
        self.init_metadata()
        parent_group = "dataset/continuous_evolution"
        cur_group = f"{self.metadata['NETWORK_NAME']}_{self.metadata['DYNAMIC_NAME']}"
        DataHandler.__init__(self, parent_group, cur_group)

    def begin(self):
        self.TIME_EVOLUTION_STEPS = self.dataset_config.TIME_EVOLUTION_STEPS
        self.DEVICE = self.config.DEVICE
        self.network = get_network(self.config)
        self.dynamics = get_dynamics(self.config)

    def _init_dataset(self):
        assert self.network.MAX_DIMENSION == self.dynamics.MAX_DIMENSION
        NUM_NODES = self.network.NUM_NODES
        NUM_STATES = self.dynamics.NUM_STATES
        TIME_EVOLUTION_STEPS = self.TIME_EVOLUTION_STEPS
        self.x0_T = torch.zeros(TIME_EVOLUTION_STEPS, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.y_ob_T = torch.zeros(TIME_EVOLUTION_STEPS, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.y_true_T = torch.zeros(TIME_EVOLUTION_STEPS, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.weight_T = torch.zeros(TIME_EVOLUTION_STEPS, NUM_NODES).to(self.config.DEVICE, dtype=torch.float)

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
        self.dynamics.init_stateof_network()  # 在T_INIT时间后重置网络状态
        # 生成数据集
        for t in range(self.TIME_EVOLUTION_STEPS):
            # 动力学初始化
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

