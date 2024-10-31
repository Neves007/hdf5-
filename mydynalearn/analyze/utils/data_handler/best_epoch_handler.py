import os
import pandas as pd
import numpy as np
from mydynalearn.logger import Log
from mydynalearn.util.lazy_loader import DataFrameLazyLoader
from mydynalearn.analyze.analyzer.exp_models_analyzer import ExpModelsAnalyzer

class BestEpochHandler(DataFrameLazyLoader):
    def __init__(self, config, exp_generator):
        self.config = config
        self.file_path = self.__get_best_epoch_dataframe_file_path()
        DataFrameLazyLoader.__init__(self, self.file_path)
        self.exp_generator = exp_generator
        self.best_epoch_dataframe = pd.DataFrame()

    def __get_best_epoch_dataframe_file_path(self):
        root_dir_path = self.config.root_dir_path
        dataframe_dir_name = self.config.dataframe_dir_name
        # 创建文件夹
        dataframe_dir_path = os.path.join(root_dir_path, dataframe_dir_name)
        if not os.path.exists(dataframe_dir_path):
            os.makedirs(dataframe_dir_path)
        # 返回文件路径
        best_epoch_dataframe_file_name = "BestEpochDataframe.csv"
        best_epoch_dataframe_file_path = os.path.join(dataframe_dir_path, best_epoch_dataframe_file_name)
        return best_epoch_dataframe_file_path


    def __add_best_epoch_result_item(self, best_epoch_exp_item):
        self.best_epoch_dataframe = pd.concat([self.best_epoch_dataframe, best_epoch_exp_item])

    def _create_data(self):
        for exp in self.exp_generator:
            # 实验模型分析器：用于分析一个实验中的所有epoch的模型
            exp_models_analyzer = ExpModelsAnalyzer(self.config, exp)
            best_epoch_exp_item, best_epoch_index = exp_models_analyzer.find_best_epoch()
            self.__add_best_epoch_result_item(best_epoch_exp_item)
        return self.best_epoch_dataframe