from mydynalearn.logger import Log
from mydynalearn.analyze.analyze_manager import *
from mydynalearn.config.yaml_config.configfile import ConfigFile


class AnalyzeManagerManager():
    # todo: 添加save和load 
    def __init__(self, exp_generator):
        """
        初始化 AnalyzeManagerManager
        :param exp_generator: 提供实验生成器的管理对象
        """
        self.config_analyze = ConfigFile.get_config_analyze()
        self.exp_generator = exp_generator
        # 初始化 Epoch 分析管理器
        self.epoch_analyze_manager = EpochAnalyzeManager(self.config_analyze, exp_generator)
        # 初始化模型分析管理器
        self.model_analyze_manager = ModelAnalyzeManager(self.config_analyze)
        # 初始化总体分析管理器
        self.overall_analyze_manager = OverallAnalyzeManager(self.config_analyze)


        # 通过职责链将各个分析模块组装起来，使用观察者模式
        self.epoch_analyze_manager.subscribe(self.model_analyze_manager)
        self.model_analyze_manager.subscribe(self.overall_analyze_manager)

    def run(self):
        """
        对所有实验进行分析
        """
        self.epoch_analyze_manager.run()