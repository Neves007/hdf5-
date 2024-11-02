import os
import pickle

import torch
import numpy as np
from Dao.dao import Dao
from abc import abstractmethod
class DataHandler():
    def __init__(self, parent_group, cur_group=''):
        self.parent_group = parent_group  # 父分组的路径
        self.cur_group = cur_group  # 当前组名
        self.dao = Dao(parent_group, cur_group)  # Dao对象用于数据存储和管理
        self.init_metadata()


    def set_metadata(self, metadata):
        self.metadata = metadata

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_metadata(self):
        if not hasattr(self, 'metadata'):
            if self.cur_group in self.dao.f:
                metadata = self.dao.get_metadata()
            else:
                self.build_dataset()
                self.save()
                metadata = self.metadata
        else:
            metadata = self.metadata
        self.metadata = metadata
        self.meta_reflection()
        return metadata

    def get_dataset(self):
        if not hasattr(self, 'dataset'):
            if len(self.dao.f[self.cur_group]) > 0:
                dataset = self.dao.get_dataset()
                self.dataset = self.convert_from_storage_format(dataset)
                return self.dataset
            else:
                self.build_dataset()
                self.save()
                return self.dataset
        else:
            return self.dataset


    def get_build_necessity(self):
        """
        检查是否需要重新构建数据集
        :return: 如果需要构建数据集则返回True，否则返回False
        """
        assert hasattr(self, "metadata")  # 确保元数据存在
        print("Checking if dataset needs to be rebuilt...")
        # 检查数据文件是否存在
        if not os.path.exists(self.dao.data_file):
            return True
        elif self.dao.get_metadata()==None:
            return True
        # 检查存储的元数据是否与当前对象的元数据匹配
        elif self.dao.get_metadata() != self.metadata:
            del self.dao.f[self.dao.group_path]
            print("Metadata mismatch, rebuilding required.")
            return True  # 不匹配时，需要重新构建
        else:
            print("Dataset is up-to-date, no rebuild needed.")
            return False

    def save(self):
        """
        存储数据集和元数据
        """
        # 确保self.cur_group不为空且对象包含dataset和metadata属性
        assert (self.cur_group != '' and hasattr(self, 'dataset') and
                hasattr(self, 'metadata')), "self.cur_group不为空且对象包含dataset和metadata属性."
        print(f"Saving dataset and metadata for {self.cur_group}...")

        # 使用Dao对象保存数据集和元数据
        self.convert_to_storage_format(self.dataset)
        self.dao.set_dataset(self.dataset)
        self.dao.set_metadata(self.metadata)
        self.dao.save()  # 执行保存操作
        print("Data saved successfully.")

    def load(self):
        """
        加载数据集和元数据，并执行元反射操作
        """
        print(f"Loading dataset and metadata for {self.cur_group}...")
        self.dataset = self.get_dataset()  # 从Dao对象加载数据集
        self.metadata = self.get_metadata()  # 从Dao对象加载元数据

        print("Data loaded and attributes updated via meta reflection.")

    def meta_reflection(self):
        """
        将元数据中的键值对映射为对象的属性
        """
        print("Reflecting metadata to object attributes...")
        for key, value in self.metadata.items():
            setattr(self, key, value)
        print("Meta reflection complete.")

    def __str__(self):
        """
        返回对象的字符串表示，包含元数据和数据集信息
        """
        metadata_str = ", ".join([f"{key}: {value}" for key, value in self.get_metadata().items()])
        return f"DataHandler(current_group = '{self.cur_group}', Metadata = [{metadata_str}])"

    def torch_to_np(self,value):
        """将值转换为存储格式"""
        return value.cpu().numpy()  # 将tensor从GPU移回CPU并转换为numpy数组

    def np_to_torch(self,value,device='cpu'):
        """将值转换回原始格式"""
        return torch.tensor(value, device=device)  # 将NumPy数组转换为CUDA tensor

    def convert_to_storage_format(self, dataset):
        for key, value in dataset.items():
            if isinstance(value, torch.Tensor):
                dataset[key] = value.detach().cpu().numpy()
            elif isinstance(value, dict):
                serialized_value = pickle.dumps(value)
                dataset[key] = serialized_value

    @abstractmethod
    def convert_from_storage_format(self, dataset):
        pass
    @classmethod
    def build_dataset(self):
        pass
    @abstractmethod
    def init_metadata(self):
        pass