# 获取配置
from mydynalearn.config import ConfigExp
from .experiment import Experiment
from mydynalearn.analyze import AnalyzeManagerManager
from mydynalearn.logger import Log
from mydynalearn.util.params_dealer import ParamsDealer
from Dao import Dao

class ExperimentManager():
    def __init__(self,params_exp_dict, fix_config):
        self.params_exp_dict = params_exp_dict
        params_dealer = ParamsDealer(params=self.params_exp_dict)
        self.params_exp = params_dealer.get_parse_params()
        self.fix_config = fix_config
        self.analyze_manager = AnalyzeManagerManager(self.exp_generator)
        self.logger = Log("ExperimentManager")
        self.TASKS = [
            "run_experiment",
            "analyze_result",
        ]

    def analyze_result(self):
        '''
        分析结果
        :return:
        '''
        self.analyze_manager.run()

    def run_experiment(self):
        '''
        运行所有实验
        '''
        # 跑实验数据
        for exp in self.exp_generator:
            exp.run()

    @property
    def exp_generator(self):
        '''
        实验生成器
        :return:
        '''
        for params in self.params_exp:
            config_exp = ConfigExp(*params)
            config_exp.do_fix_config(self.fix_config)
            exp = Experiment(config_exp)
            yield exp



    def run(self):
        """
        运行实验任务
        """
        with Dao() as dao:
            for task_name in self.TASKS:
                task_method = getattr(self, task_name, None)
                if callable(task_method):
                    self.logger.increase_indent()
                    self.logger.log(f"Do exp task, name = {task_name}.")
                    task_method()
                    self.logger.decrease_indent()
                else:
                    raise ValueError(f"{task_name} is an invalid task, possible tasks are {self.TASKS}")
