import h5py
import numpy as np


class Dao:
    f = None  # 全局静态属性，用于存储打开的 HDF5 文件

    def __init__(self, parent_group=".", cur_group_name="dataset"):
        """
        初始化 Dao 对象，设置文件路径、父组和当前组的名称，并生成组的绝对路径。

        参数：
        - parent_group: 父组的名称。
        - cur_group_name: 当前组的名称。
        """
        self.data_file = "output/dataset.h5"  # 数据文件名称
        self.parent_group = parent_group  # 父组的名称
        self.cur_group = cur_group_name  # 当前组的名称
        self.group_path = self.get_group_path()  # 生成组的绝对路径

    def set_metadata(self, metadata):
        """
        设置元数据。

        参数：
        - metadata: 一个字典，包含需要存储的元数据。
        """
        self.metadata = metadata  # 将元数据赋值给对象属性

    def set_dataset(self, dataset):
        """
        设置数据集。

        参数：
        - dataset: 一个字典，包含需要存储的数据集。
        """
        self.dataset = dataset  # 将数据集赋值给对象属性

    def get_metadata(self):
        """
        获取元数据。如果元数据尚未加载，则从文件中读取并赋值给对象属性。

        返回：
        - metadata: 一个字典，包含元数据。
        """
        if not hasattr(self, 'metadata'):
            if self.group_path in Dao.f:
                # 从文件中加载元数据并赋值给 self.metadata
                self.metadata = {name: Dao.f[self.group_path].attrs[name] for name in Dao.f[self.group_path].attrs}
            else:
                print(f"No metadata found for '{self.group_path}' in the file.")
                return None
        return self.metadata

    def get_dataset(self):
        """
        获取数据集。如果数据集尚未加载，则从文件中读取并赋值给对象属性。

        返回：
        - dataset: 一个字典，包含数据集。
        """
        if not hasattr(self, 'dataset'):
            if self.group_path in Dao.f:
                # 从文件中加载数据集并赋值给 self.dataset
                self.dataset = {name: Dao.f[self.group_path][name][...] for name in Dao.f[self.group_path].keys()}
            else:
                print(f"No dataset found for '{self.group_path}' in the file.")
                return None
        return self.dataset

    def compare_data_structure(self, target_dataset):
        """
        比较 target_dataset 的键和文件中 group 的键是否相同
        :param target_dataset: 用于比较的目标数据集
        :return: 如果键相同则返回 True，否则返回 False
        """
        if self.group_path in Dao.f:
            group = Dao.f[self.group_path]
            group_keys = list(group.keys())  # 获取文件中 group 的键
            target_keys = list(target_dataset.keys())  # 获取 target_dataset 的键

            # 比较两个键列表是否相同
            if group_keys == target_keys:
                print("Group keys match with target dataset keys.")
                return True
            else:
                print("Group keys do not match with target dataset keys.")
                return False
        else:
            print(f"No dataset found for '{self.group_path}' in the file.")
            return False

    def get_group_path(self):
        """
        生成并返回当前组的绝对路径。

        返回：
        - group_path: 当前组的绝对路径字符串。
        """
        return f"{self.parent_group}/{self.cur_group}"

    def save(self):
        """
        将 dataset 和 metadata 保存到 HDF5 文件中。

        要求 dataset 和 metadata 属性已存在，否则抛出异常。
        """
        assert (hasattr(self, 'dataset') and hasattr(self,
                                                     'metadata')), "Both 'dataset' and 'metadata' attributes must exist."

        # 检查目标路径是否已存在
        if self.group_path not in Dao.f:
            # 创建一个新的组并存储 dataset 和 metadata
            data_group = Dao.f.create_group(self.group_path)

            # 将 dataset 存储到组中
            for name, data in self.dataset.items():
                data_group.create_dataset(name, data=data)

                # 将 metadata 存储为组的属性
            for name, data in self.metadata.items():
                data_group.attrs[name] = data

            print(f"Dataset '{self.group_path}' created and stored.")
        else:
            print(f"Dataset '{self.group_path}' already exists, skipping creation.")

    def __enter__(self):
        if Dao.f is None:
            Dao.f = h5py.File(self.data_file, "a")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if Dao.f is not None:
            Dao.f.close()
            Dao.f = None