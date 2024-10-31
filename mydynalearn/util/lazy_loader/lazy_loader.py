import os
from abc import ABC, abstractmethod


class LazyLoader(ABC):
    def __init__(self, data_file):
        """初始化 LazyLoader 实例。

        Args:
            data_file (str): 数据文件的路径。
        """
        self.data_file = data_file
        self.data = None

    def load_data(self):
        """加载数据，如果数据不存在则创建并保存。
        """
        if self.data is None:
            if self.data_file_exist():

                self.data = self._read_from_file()
            else:
                self.data = self._create_data()
                self._save_data()

    def get_data(self):
        """获取数据，确保数据已被加载。
        """
        if self.data is None:
            self.load_data()
        return self.data

    def ensure_data_file_exists(self):
        """确保数据文件存在。

        如果数据文件存在，则无需进行任何操作；
        如果不存在，则创建数据并保存到文件。
        """
        if not self.data_file_exist():
            self.data = self._create_data()
            self._save_data()

    def data_file_exist(self):
        """检查数据文件是否存在。

        Returns:
            bool: 如果数据文件存在，则返回 True；否则返回 False。
        """
        return os.path.exists(self.data_file)

    @abstractmethod
    def _read_from_file(self):
        """从文件中读取数据。

        子类必须实现该方法以定义如何从特定文件格式读取数据。

        Returns:
            数据：从文件中读取的数据。
        """
        pass

    @abstractmethod
    def _save_data(self):
        """将数据保存到文件中。

        子类必须实现该方法以定义如何将数据保存到特定文件格式。
        """
        pass


    def _create_data(self):
        """创建新的数据。

        子类必须实现该方法以定义如何生成新的数据。

        Returns:
            数据：新创建的数据。
        """
        pass
