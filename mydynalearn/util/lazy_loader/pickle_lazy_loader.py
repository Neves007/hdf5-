import pickle
from .lazy_loader import LazyLoader
from abc import abstractmethod

class PickleLazyLoader(LazyLoader):
    def _read_from_file(self):
        """使用 pickle 从文件中读取数据。"""
        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"读取文件 {self.data_file} 时出错: {e}")
            return None

    def _save_data(self):
        """使用 pickle 将数据保存到文件中。"""
        try:
            with open(self.data_file, 'wb') as f:
                pickle.dump(self.data, f)
        except Exception as e:
            print(f"保存数据到 {self.data_file} 时出错: {e}")

    @abstractmethod
    def _create_data(self):
        """创建新的数据。

        子类必须实现该方法以定义如何生成新的数据。

        Returns:
            数据：新创建的数据。
        """
        return self.data