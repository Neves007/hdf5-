import os
import torch.nn as nn
from mydynalearn.model.optimizer import get as get_optimizer

from mydynalearn.model.epoch_tasks import EpochTasks
from mydynalearn.logger import Log

from torch.utils.data import DataLoader
import torch
class Model():
    def __init__(self, config):
        """Dense version of GAT."""
        # config
        self.config = config
        self.logger = Log("Model")
        self.NAME = config.model.NAME
        self.epoch_tasks = EpochTasks(config)
        self.need_to_train = self.epoch_tasks.need_to_train

    # 放进数据集类里面

    # 定义模型
    def run(self,dataset):
        self.logger.increase_indent()
        self.logger.log("Beginning model training...")
        self.epoch_tasks.run_all(**dataset)
        self.logger.decrease_indent()