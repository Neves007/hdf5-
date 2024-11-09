from Dao import Dao
from mydynalearn.dataset import *
from mydynalearn.model import Model
from mydynalearn.logger import Log


class Experiment:
    def __init__(self, config):
        self.config = config
        self.logger = Log("Experiment")
        self.NAME = config.NAME
        self.network = get_network(self.config)
        self.periodic_dateset = PeriodicDateset(self.config)
        self.continuous_dataset = ContinuousDataset(self.config)
        self.split_dataset = SplitDataset(self.config)
        self.model = Model(config, self.split_dataset)
        self.TASKS = [
            "create_network",
            "create_dynamic_dataset",
            "split_dynamic_dataset",
            "train_model",
        ]

    def create_network(self):
        self.network.run()

    def create_dynamic_dataset(self):
        """
        创建常规的动力学数据集
        """
        self.periodic_dateset.run()
        self.continuous_dataset.run()

    def split_dynamic_dataset(self):
        self.split_dataset.run()

    def train_model(self):
        """
        训练模型，如果需要训练
        """
        self.model.run()

    def run(self):
        """
        运行实验任务
        """
        self.logger.increase_indent()
        self.logger.log(f"train experiment: {self.NAME}")

        for task_name in self.TASKS:
            task_method = getattr(self, task_name, None)
            if callable(task_method):
                self.logger.increase_indent()
                self.logger.log(f"Do exp task, name = {task_name}.")
                task_method()
                self.logger.decrease_indent()
            else:
                raise ValueError(f"{task_name} is an invalid task, possible tasks are {self.TASKS}")
        self.logger.decrease_indent()

