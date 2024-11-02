# 获取配置
from mydynalearn.config import ConfigExp
from .experiment import Experiment
from mydynalearn.logger import Log
from mydynalearn.util.params_dealer import ParamsDealer


class ExperimentManager():
    def __init__(self,params_exp_dict, fix_config):
        self.params_exp_dict = params_exp_dict
        params_dealer = ParamsDealer(params=self.params_exp_dict)
        self.params_exp = params_dealer.get_parse_params()
        self.fix_config = fix_config
        self.logger = Log("ExperimentManager")

    def get_exp_generator(self):
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
        '''
        训练模型
        输出：模型的参数 model_state_dict
        '''

        self.logger.log("TRAINING PROCESS")
        exp_generator = self.get_exp_generator()
        for exp in exp_generator:
            exp.run()
            # torch.cuda.empty_cache()
        self.logger.log("TRAINING PROCESS COMPLETED!\n\n")


