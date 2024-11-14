import h5py
import numpy as np


class Dao:
    f = None  # 全局静态属性，用于存储打开的 HDF5 文件
    data_file = None  # 全局静态属性，用于存储数据文件路径

    def __init__(self, parent_group=".", cur_group_name="dataset", dao_name=None):
        """
        初始化 Dao 对象，设置文件路径、父组和当前组的名称，并生成组的绝对路径。

        参数：
        - parent_group: 父组的名称。
        - cur_group_name: 当前组的名称。
        - dao_name: 数据文件名称（仅在首次实例化时需要传入）。
        """
        if Dao.data_file is None and dao_name is not None:
            Dao.data_file = f"output/{dao_name}.h5"  # 仅在首次实例化时设置 data_file

        if Dao.data_file is None:
            raise ValueError("Dao.data_file 未设置。首次实例化时需要提供 dao_name 参数。")

        self.parent_group = parent_group  # 父组的名称
        cur_group = cur_group_name  # 当前组的名称
        self.group_path = self.set_group_path(cur_group)  # 生成组的绝对路径



    def get_group_obj(self):
        if self.group_path in self.f:
            group_obj = self.f[self.group_path]
        else:
            group_obj = None
        return group_obj

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
                return None
        return self.metadata

    def get_dataset(self):
        """
        获取数据集。如果数据集尚未加载，则从文件中读取并赋值给对象属性。

        返回：
        - dataset: 一个字典，包含数据集。
        """
        if not hasattr(self, 'dataset'):

            if self.group_path in self.f:
                self.dataset = {}
                for name in self.f[self.group_path].keys():
                    if isinstance(self.f[self.group_path][name], h5py.Dataset):
                        self.dataset[name] = self.f[self.group_path][name][...]

                # 从文件中加载数据集并赋值给 self.dataset

            else:
                return None
        return self.dataset

    def set_group_path(self,cur_group):
        """
        生成并返回当前组的绝对路径。

        返回：
        - group_path: 当前组的绝对路径字符串。
        """
        return f"{self.parent_group}/{cur_group}"

    def save(self):
        """
        将 dataset 和 metadata 保存到 HDF5 文件中。

        要求 dataset 和 metadata 属性已存在，否则抛出异常。
        """
        assert (hasattr(self, 'dataset') and hasattr(self,
                                                     'metadata')), "Both 'dataset' and 'metadata' attributes must exist."

        # 检查目标路径是否已存在
        if self.group_path not in Dao.f:
            try:
                # 创建一个新的组并存储 dataset 和 metadata
                data_group = Dao.f.create_group(self.group_path)

                # 将 dataset 存储到组中
                for name, data in self.dataset.items():
                    data_group.create_dataset(name, data=data)

                    # 将 metadata 存储为组的属性
                for name, data in self.metadata.items():
                    data_group.attrs[name] = data
            except Exception as e:
                # 发生错误时删除已创建的分组
                if self.group_path in Dao.f:
                    del Dao.f[self.group_path]
                print(f"An error occurred: {e}. Group '{self.group_path}' has been removed.")
                raise e  # 重新抛出异常
        else:
            raise FileExistsError(f"Dataset '{self.group_path}' already exists.")


    def __enter__(self):
        if Dao.f is None:
            Dao.f = h5py.File(Dao.data_file, "a")  # 仅在首次调用时打开文件
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if Dao.f is not None:
            Dao.f.close()
            Dao.f = None
