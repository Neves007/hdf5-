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
        self.epoch_analyze_manager = EpochAnalyzeManager(self.config_analyze, exp_generator)
        self.logger = Log("AnalyzeManagerManager")
        self.TASKS = [
            "analyze_every_epoch",
            "analyze_every_exp",
            "analyze_overall"
        ]

    def analyze_every_epoch(self):
        '''
        分析每个epoch
        :return:
        '''
        self.epoch_analyze_manager.run()
        pass
    def analyze_every_exp(self):
        '''
        分析每个实验的所有epoch
        :return:
        '''
        pass
    def analyze_overall(self):
        '''
        分析所有实验
        :return:
        '''
        pass

    def run(self):
        """
        对所有实验进行分析
        """
        for task_name in self.TASKS:
            task_method = getattr(self, task_name, None)
            if callable(task_method):
                self.logger.increase_indent()
                self.logger.log(f"Do exp task, name = {task_name}.")
                task_method()
                self.logger.decrease_indent()
            else:
                raise ValueError(f"{task_name} is an invalid task, possible tasks are {self.TASKS}")

