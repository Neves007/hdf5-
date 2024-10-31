import pandas as pd
from .lazy_loader import LazyLoader
from abc import abstractmethod

class DataFrameLazyLoader(LazyLoader):
    def _read_from_file(self):
        """从文件中读取 DataFrame。"""
        try:
            if self.data_file.endswith('.csv'):
                data = pd.read_csv(self.data_file)
            elif self.data_file.endswith('.xlsx'):
                data = pd.read_excel(self.data_file)
            elif self.data_file.endswith('.pkl'):
                data = pd.read_pickle(self.data_file)
            else:
                raise ValueError("不支持的文件格式。")
            return data
        except Exception as e:
            print(f"读取文件 {self.data_file} 时出错: {e}")
            return None

    def _save_data(self):
        """将 DataFrame 保存到文件中。"""
        try:
            if self.data_file.endswith('.csv'):
                self.data.to_csv(self.data_file, index=False)
            elif self.data_file.endswith('.xlsx'):
                self.data.to_excel(self.data_file, index=False)
            elif self.data_file.endswith('.pkl'):
                self.data.to_pickle(self.data_file)
            else:
                raise ValueError("不支持的文件格式。")
        except Exception as e:
            print(f"保存数据到 {self.data_file} 时出错: {e}")

    @abstractmethod
    def _create_data(self):
        """创建新的数据。

        子类必须实现该方法以定义如何生成新的数据。

        Returns:
            数据：新创建的数据。
        """
        pass