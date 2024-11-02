from Dao import Dao
from mydynalearn.dataset import *
from mydynalearn.model import Model
from mydynalearn.logger import Log


class Experiment:
    def __init__(self, config):
        self.config = config
        self.logger = Log("Experiment")
        self.exp_info = self._get_exp_info()
        self.NAME = config.NAME
        self.network = get_network(self.config)
        self.dynamics = get_dynamics(self.config)
        self.dataset = PeriodicDateset(self.config)
        self.model = Model(config)
        self.TASKS = [
            "create_dynamic_dataset",
            "train_model",
        ]

    def create_dynamic_dataset(self):
        """
        创建常规的动力学数据集
        """
        self.dataset.run()

    def train_model(self):
        """
        训练模型，如果需要训练
        """
        if self.model.need_to_train:
            dataset = self.dataset.get_data()
            self.model.run(dataset)
        else:
            self.logger.log("The model has already been trained!")

    def _get_exp_info(self):
        """
        获取全局信息，包括模型和数据集的相关配置
        """
        return {
            "model_network_name": self.config.network.NAME,
            "model_dynamics_name": self.config.dynamics.NAME,
            "dataset_network_name": self.config.network.NAME,
            "dataset_dynamics_name": self.config.dynamics.NAME,
            "model_name": self.config.model.NAME,
        }


    def run(self):
        """
        运行实验任务
        """
        self.logger.increase_indent()
        self.logger.log(f"train experiment: {self.NAME}")
        with Dao() as dao:
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

