import os
import pandas as pd
import numpy as np
from mydynalearn.logger import Log
from mydynalearn.util.lazy_loader import DataFrameLazyLoader


class ExpBestEpochHandler(DataFrameLazyLoader):
    def __init__(self, config, exp, model_performance_analyze_result_generator):
        self.config = config
        self.exp = exp
        self.file_path = self.__get_exp_dataframe_file_path()
        DataFrameLazyLoader.__init__(self, self.file_path)
        self.model_performance_analyze_result_generator = model_performance_analyze_result_generator
        self.exp_dataframe = pd.DataFrame()

    def __get_exp_dataframe_file_path(self):
        root_dir_path = self.config.root_dir_path
        dataframe_dir_name = self.config.dataframe_dir_name
        # 创建文件夹
        dataframe_dir_path = os.path.join(root_dir_path, dataframe_dir_name)
        if not os.path.exists(dataframe_dir_path):
            os.makedirs(dataframe_dir_path)
        # 返回文件路径
        exp_dataframe_file_name = '_'.join([self.exp.NAME, "ExpDataframe.csv"])
        exp_dataframe_file_path = os.path.join(dataframe_dir_path, exp_dataframe_file_name)
        return exp_dataframe_file_path

    def __analyze_result_to_pdseries(self, analyze_result):
        test_result_info = analyze_result['test_result_info']
        model_performance_dict = analyze_result['model_performance_dict']
        performance_dict = {"f1": model_performance_dict["f1"],
                            "R": model_performance_dict["R"],
                            "cross_loss": model_performance_dict["cross_loss"], }
        analyze_result_series = pd.Series({**test_result_info, **performance_dict})
        return analyze_result_series

    def __add_epoch_result_item(self, analyze_result):
        analyze_result_series_for_a_epoch = self.__analyze_result_to_pdseries(analyze_result)
        self.exp_dataframe = pd.concat([self.exp_dataframe, analyze_result_series_for_a_epoch.to_frame().T])

    def _create_data(self):
        for model_performance_analyze_result in self.model_performance_analyze_result_generator:
            self.__add_epoch_result_item(model_performance_analyze_result)
        return self.exp_dataframe

    def find_best_epoch(self):
        self.exp_dataframe = self.get_data()
        # 找到 cross_loss 最小值对应的行
        # Assuming self.exp_dataframe is your DataFrame
        self.exp_dataframe['cross_loss'] = pd.to_numeric(self.exp_dataframe['cross_loss'], errors='coerce')
        # 找到 cross_loss 列最小的值和对应的行索引
        min_cross_loss_item = self.exp_dataframe.nsmallest(1, 'cross_loss').reset_index(drop=True)

        # 提取该行中对应的 model_epoch_index 值，并转换为整数
        # 提取 model_epoch_index 的值
        value = min_cross_loss_item.at[0, 'model_epoch_index']

        # value有时是int有时是series。处理数据
        if isinstance(value, int):
            min_cross_loss_epoch_index = value
        else:
            # 如果不是 int 类型，假设是 Series 或其他类型，则调用 .item()
            min_cross_loss_epoch_index = value.item()
        return min_cross_loss_item, min_cross_loss_epoch_index