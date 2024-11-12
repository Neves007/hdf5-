import os
import torch.nn as nn
from mydynalearn.model.optimizer import get as get_optimizer

from mydynalearn.model.epoch_tasks import EpochTasks
from mydynalearn.logger import Log

from torch.utils.data import DataLoader
import torch
class Model():
    def __init__(self, config, split_dataset):
        """Dense version of GAT."""
        # config
        self.config = config
        self.logger = Log("Model")
        self.NAME = config.model.NAME
        self.EPOCHS = config.model.EPOCHS

    # 定义模型
    def run(self):
        self.logger.increase_indent()
        self.logger.log("Beginning model training...")
        for epoch_tasks in self.epoch_task_generator():
            epoch_tasks.run()
            epoch_tasks.low_the_lr()
        self.logger.decrease_indent()

    def epoch_task_generator(self):
        for epoch_index in range(self.EPOCHS):
            epoch_task = self.get_epoch_task(epoch_index)
            yield epoch_task

    def get_epoch_task(self, epoch_index):
        epoch_tasks = EpochTasks(self.config, epoch_index)
        return epoch_tasks