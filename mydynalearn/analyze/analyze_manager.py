from mydynalearn.analyze.analyzer.exp_models_analyzer import ExpModelsAnalyzer
from mydynalearn.config import Config
import os
import pandas as pd
from mydynalearn.logger import Log
from mydynalearn.util.lazy_loader import DataFrameLazyLoader
from mydynalearn.analyze.utils.data_handler.best_epoch_handler import BestEpochHandler
class AnalyzeManager():
    # todo: 添加save和load 
    def __init__(self, exp_generator, indent=0):
        """
        初始化 AnalyzeManager
        :param exp_generator: 提供实验生成器的管理对象
        """
        config_analyze = Config.get_config_analyze()
        self.logger = Log("AnalyzeManager")
        self.indent=indent
        self.config = config_analyze['default']
        self.exp_generator = exp_generator
        self.best_epoch_handler = BestEpochHandler(self.config,exp_generator)

    def main_analyze_exp(self):
        """
        对单个实验中的所有epoch的模型进行分析
        :param exp: 实验对象
        """
        for exp in self.exp_generator:
            # 实验模型分析器：用于分析一个实验中的所有epoch的模型
            exp_models_analyzer = ExpModelsAnalyzer(self.config, exp)
            exp_models_analyzer.run()

    def main_buid_best_epoch_dataframe(self):
        '''
        处理best_epoch_dataframe
        :return:
        '''
        self.best_epoch_handler.ensure_data_file_exists()

    def run(self):
        """
        对所有实验进行分析
        """
        # 分析每一个实验
        self.logger.log("start analyzing...")
        self.logger.increase_indent()
        self.logger.log("analyze experiments...")
        self.main_analyze_exp()
        # 处理best_epoch_dataframe
        self.logger.log(f"buid best epoch dataframe")
        self.main_buid_best_epoch_dataframe()
        self.logger.decrease_indent()
        self.logger.log("end")
        print("\n")
