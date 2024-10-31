import os.path
import pickle
import torch
from tqdm import *
from abc import abstractmethod
from mydynalearn.networks import *
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics
from torch.utils.data import Dataset, Subset
from mydynalearn.logger import Log
from mydynalearn.util.lazy_loader import *
from .dynamic_dataset_time_evolution_ori import DynamicDatasetTimeEvolutionOrigion
from .dynamic_dataset_time_evolution_ml import DynamicDatasetTimeEvolutionML

class DynamicDatasetTimeEvolution(PickleLazyLoader):
    '''数据集类
    通过网络network和dynamics来说生成动力学数据集

    生成数据：
        - run_dynamic_process 连续时间动力学数据
        - run 生成动力学数据
    '''
    def __init__(self, exp, ml_model, network, dynamics) -> None:
        self.config = exp.config
        self.logger = Log("DynamicDatasetTimeEvolution")
        self.exp = exp
        self.ml_model = ml_model
        self.network = network
        self.dynamics = dynamics
        self.dataset_file_path = self._get_dataset_file_path()
        super().__init__(self.dataset_file_path)
        self.origion_time_evolution = DynamicDatasetTimeEvolutionOrigion(exp, ml_model, network, dynamics)
        self.ml_time_evolution = DynamicDatasetTimeEvolutionML(exp, ml_model, network, dynamics)


    def _get_dataset_file_path(self):
        dataset_file_name = f"TIME_EVOLUTION_DATASET_{self.network.NAME}_{self.dynamics.NAME}_{self.exp.model.NAME}_epoch{self.ml_model.epoch_index}.pkl"
        return os.path.join(self.config.time_evolution_dataset_dir_path, dataset_file_name)


    def _create_data(self):
        """
        构建新的数据集并保存到文件中。
        """
        # 构建原始动力学模型的数据集
        self.logger.increase_indent()
        self.logger.log("build and save time evolution dataset")
        origion_time_evolution_data = self.origion_time_evolution.get_data()
        ml_time_evoluion_data = self.ml_time_evolution.get_data()
        data = {
            "ori_x0_T": origion_time_evolution_data['data'],
            "ml_x0_T": ml_time_evoluion_data['data'],
        }
        self.logger.decrease_indent()
        return data