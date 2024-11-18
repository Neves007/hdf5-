import os
import torch.nn as nn
from mydynalearn.model.optimizer import get as get_optimizer

from mydynalearn.model.epoch_tasks import EpochTasks
from mydynalearn.logger import Log
from mydynalearn.model.getter import get as get_model
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
        # Initialize model, optimizer, and scheduler once
        self.model = self.initialize_model()
        self.optimizer = get_optimizer(config)(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=4, eps=1e-8, threshold=0.1
        )

    def initialize_model(self):
        # Initialize and return the model instance
        return get_model(self.config)

    # Define the model run method
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
        # Instead of creating a new model instance every time, pass the existing model and optimizer
        epoch_tasks = EpochTasks(self.config, epoch_index, model=self.model, optimizer=self.optimizer,
                                 scheduler=self.scheduler)
        return epoch_tasks