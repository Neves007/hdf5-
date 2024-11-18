# 获取配置
from mydynalearn.config import ConfigExp
from mydynalearn.experiments.experiment import Experiment
from mydynalearn.logger import Log
from ..analyze import Analyze
from ..visualize import visualizerManager

from Dao import Dao
import os

class ExperimentManager():
    def __init__(self,params_exp):
        os.makedirs("output", exist_ok=True)
        self.params_exp = params_exp
        self.analyze = Analyze(self.exp_generator)
        self.logger = Log("ExperimentManager")
        self.TASKS = [
            "run_experiment",
            "analyze_result",
            "visualize"
        ]

    def run_experiment(self):
        '''
        运行所有实验
        '''
        # 跑实验数据
        for exp in self.exp_generator():
            exp.run()

    def analyze_result(self):
        '''
        分析结果
        :return:
        '''
        self.analyze.run()

    def visualize(self):
        visualizer_manager = visualizerManager(self.analyze.get_output())
        visualizer_manager.run()


    def exp_generator(self):
        '''
        实验生成器
        :return:
        '''
        for params in self.params_exp:
            config_exp = ConfigExp(**params)
            exp = Experiment(config_exp)
            yield exp



    def run(self, data_base):
        """
        运行实验任务
        """
        with Dao(dao_name = data_base) as dao:
            for task_name in self.TASKS:
                task_method = getattr(self, task_name, None)
                if callable(task_method):
                    self.logger.increase_indent()
                    self.logger.log(f"Do exp task, name = {task_name}.")
                    task_method()
                    self.logger.decrease_indent()
                else:
                    raise ValueError(f"{task_name} is an invalid task, possible tasks are {self.TASKS}")
