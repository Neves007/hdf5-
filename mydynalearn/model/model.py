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
        self.EPOCHS = config.model.EPOCHS

    # 放进数据集类里面

    # 定义模型
    def run(self,dataset):
        self.logger.increase_indent()
        self.logger.log("Beginning model training...")
        self.epoch_tasks.run_all(**dataset)
        self.logger.decrease_indent()

    def run_all_epochs(self,dataset):
        self.logger.increase_indent()
        self.logger.log("Beginning model training...")
        for epoch_index in range(self.EPOCHS):
            epoch_tasks = EpochTasks(self.config, epoch_index)
            if epoch_tasks.get_build_necessity():
                epoch_tasks.build_dataset()
                epoch_tasks.save()
                epoch_tasks.low_the_lr(epoch_index)
        self.logger.decrease_indent()

