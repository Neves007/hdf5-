from abc import ABC, abstractmethod
import torch

# 抽象基类，用于定义实验的基本结构
class Replication(ABC):
    def __init__(self, replication_data_name_list=None, n=1):
        self.n = n  # 实验重复运行的次数
        self.replication_data_name_list = replication_data_name_list  # 实验数据的键名列表

    def run_replication(self):
        self._create_trial_data()  # 生成首次实验数据
        self.replication_data = self._get_replication_data_item()  # 获取指定实验数据

        # 运行剩余 n-1 次实验并累加结果
        for _ in range(1, self.n):
            self._create_trial_data()
            self._add_replication_data_item(self._get_replication_data_item())

        return self._do_mean()  # 计算并返回平均值

    def _get_replication_data_item(self):
        # 从实验数据中提取指定的数据项
        return {key: self.data[key] for key in self.replication_data_name_list if key in self.data}

    def _add_replication_data_item(self, replication_data_item):
        # 累加新的实验数据项到现有数据中
        for key in self.replication_data:
            self.replication_data[key] += replication_data_item[key]

    def _do_mean(self):
        # 计算实验数据的平均值
        return {key: value / self.n for key, value in self.replication_data.items()}

    @abstractmethod
    def _create_trial_data(self):
        pass