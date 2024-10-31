import os.path
import pickle
import random
from abc import abstractmethod
from mydynalearn.logger import Log
from Dao.data_handler import DataHandler

class Network(DataHandler):
    def __init__(self, config):
        # toy_network = True
        self.logger = Log("Network")
        self.config = config
        self.DEVICE = self.config.DEVICE
        metadata =  config.network
        metadata["DEVICE"] = self.DEVICE
        self.set_metadata(metadata)
        parent_group = "network"  # 父分组的路径
        cur_group = config.NAME  # 当前网络的组名
        super().__init__(parent_group=parent_group, cur_group=cur_group)
    
    def _create_data(self):
        """创建新的数据。

        子类必须实现该方法以定义如何生成新的数据。

        Returns:
            数据：新创建的数据。
        """
        self.build()
        self.to_device(self.DEVICE)
        return self

    def run(self):
        if self.get_build_necessity():
            self.build()
            self.save()


    @abstractmethod
    def _update_adj(self):
        pass

    @abstractmethod
    def to_device(self,device):
        pass

    def convert_to_storage_format(self, dataset):
        """
        将值转换为存储格式，子类可以重写此方法以自定义行为
        """
        att_list = ["nodes","edges","inc_matrix_adj0","inc_matrix_adj1"]
        for att in att_list:
            dataset[att] = self.torch_to_np(dataset[att])
        return dataset

    def convert_from_storage_format(self, dataset):
        """
        将值转换回原始格式，子类可以重写此方法以自定义行为
        """
        att_list = ["nodes","edges","inc_matrix_adj0","inc_matrix_adj1"]
        for att in att_list:
            dataset[att] = self.np_to_torch(dataset[att], device=self.DEVICE)
        return dataset