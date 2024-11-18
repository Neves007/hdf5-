import os
import pickle

import torch
import numpy as np
from sympy.stats.sampling.sample_numpy import numpy
import h5py

from Dao.dao import Dao
from abc import abstractmethod
from mydynalearn.logger import Log
class DataHandler():
    dao_name = None
    def __init__(self, parent_group, cur_group=''):
        self.parent_group = parent_group  # 父分组的路径
        self.cur_group = cur_group  # 当前组名
        self.dao = Dao(parent_group, cur_group, self.dao_name)  # Dao对象用于数据存储和管理
        self.init_metadata()
        self.log = Log("DataHandler")


    def set_metadata(self, metadata):
        self.metadata = metadata
        self.__meta_reflection()

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.__dataset_reflection()

    def get_metadata(self):
        return self.dao.get_metadata()

    def get_dataset(self):
        if not hasattr(self, 'dataset'):
            dao_group_obj = self.dao.get_group_obj()
            if dao_group_obj == None:
                self.run()
            else:
                dataset = self.dao.get_dataset()
                self.dataset = self.convert_from_storage_format(dataset)
        self.__dataset_reflection()
        return self.dataset


    def _get_build_necessity(self):
        """
        检查是否需要重新构建数据集
        :return: 如果需要构建数据集则返回True，否则返回False
        """
        # 检查数据文件是否存在
        if not os.path.exists(self.dao.data_file):
            return True
        # 组是否存在
        elif self.dao.group_path not in Dao.f:
            return True
        # meta数据是否存在
        elif self.dao.get_metadata()==None:
            return True
        else:
            return False

    def _save(self):
        """
        存储数据集和元数据
        """
        # 确保self.cur_group不为空且对象包含dataset和metadata属性
        assert (self.cur_group != '' and hasattr(self, 'dataset') and
                hasattr(self, 'metadata')), f"self.cur_group = {self.cur_group}"

        # 使用Dao对象保存数据集和元数据
        self.convert_to_storage_format(self.dataset)
        self.dao.set_dataset(self.dataset)
        self.dao.set_metadata(self.metadata)
        self.dao.save()  # 执行保存操作


    def load(self):
        """
        加载数据集和元数据，并执行元反射操作
        """
        self.dataset = self.get_dataset()  # 从Dao对象加载数据集

    def __meta_reflection(self):
        """
        将元数据中的键值对映射为对象的属性
        """
        for key, value in self.metadata.items():
            setattr(self, key, value)
    def __dataset_reflection(self):
        """
        将元数据中的键值对映射为对象的属性
        """
        for key, value in self.dataset.items():
            setattr(self, key, value)

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



    # 可以重写做个性化配置
    def begin(self,*args,**kwargs):
        pass

    # 可以重写做个性化配置
    def end(self):
        pass

    # 公用接口，保证数据的存在
    def run(self):
        self.log.increase_indent()
        if self._get_build_necessity():
            self.log.log("Building dataset...")
            self.begin()
            self.build_dataset()
            self.end()
            self._save()
            self.log.log(f"Dataset '{self.dao.group_path}' created and stored.")
        else:
            self.log.log(f"Dataset '{self.dao.group_path}'  already exists.")
        self.log.decrease_indent()

    # 可以重写做个性化配置
    def convert_to_storage_format(self, dataset):
        for key, value in dataset.items():
            if isinstance(value, torch.Tensor):
                dataset[key] = value.to_dense().detach().cpu().numpy()
            elif isinstance(value, dict):
                serialized_value = pickle.dumps(value)
                dataset[key] = serialized_value
            elif isinstance(value, np.ndarray):
                # 如果是 numpy 数组，检查是否为字符串类型
                if np.issubdtype(value.dtype, np.str_) or np.issubdtype(value.dtype, np.bytes_):
                    # 将 numpy 字符串类型转换为 HDF5 支持的字符串类型
                    dataset[key] = value.astype(h5py.string_dtype(encoding='utf-8'))
                else:
                    # 否则直接存储原始 numpy 数组
                    dataset[key] = value

            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                # 如果是 Python 字符串数组（list of str），转换为 numpy 字符串数组并再转换为 HDF5 支持的类型
                dataset[key] = np.array(value, dtype=h5py.string_dtype(encoding='utf-8'))

        pass

    def convert_from_storage_format(self, dataset):
        for key, value in dataset.items():
            # 检查字段是否为 numpy 数组
            if isinstance(value, np.ndarray):
                if value.shape != ():
                    # 如果是字节字符串类型，将其转换为字符串类型
                    if isinstance(value[0], bytes):
                        dataset[key] = value.astype(str)
                else:
                    dataset[key] = value.item()

        return dataset
    @classmethod
    def build_dataset(self,*args,**kwargs):
        pass
    @abstractmethod
    def init_metadata(self):
        pass