# 获取配置
from mydynalearn.config import ConfigDynamicTestingExp
from mydynalearn.experiments import ExperimentTestDynamic
from mydynalearn.logger import Log
from mydynalearn.util.params_dealer import PasramsDealer

class TestDynamicExperimentManager():
    def __init__(self, fix_config_dict, params_dict):
        self.fix_config_dict = fix_config_dict
        self.logger = Log("TestDynamicExperimentManager")
        self.params = PasramsDealer.assemble_test_dynamics_params(params_dict)
        self.root_dir = r"./output/"
    def get_train_exp(self,network, dynamics):
        '''通过参数获得实验对象

        :param network:
        :param dynamics:
        :param model:
        :param IS_WEIGHT:
        :return: exp
        '''
        exp_name = "testDynamic-" + network + "-" + dynamics
        kwargs = {
            "NAME": exp_name,
            "network": network,
            "dynamics": dynamics,
            "seed": 0,
            "root_dir": self.root_dir
        }
        config = ConfigDynamicTestingExp(**kwargs)
        self.fix_config(config)
        exp = ExperimentTestDynamic(config)
        return exp

    def get_exp_generator(self):
        for param in self.params:
            exp = self.get_train_exp(*param)
            yield exp

    def fix_config(self,config):
        '''调整配置
        'AVG_DEGREE': 10,
        'NUM_SAMPLES' : 1000,
        'EFF_BETA_LIST': torch.linspace(0,2,51),
        'MU':1,
        'DEVICE': torch.device('cuda'),
        '''
        # T总时间步
        config.dataset.NUM_SAMPLES = self.fix_config_dict['NUM_SAMPLES']

        config.EFF_BETA_LIST = self.fix_config_dict['EFF_BETA_LIST']
        config.DEVICE = self.fix_config_dict['DEVICE']
        config.ROUND = self.fix_config_dict['ROUND']


    def run(self):
        '''
        训练模型
        输出：模型的参数 model_state_dict
        '''
        self.logger.log("*"*10+" TRAINING PROCESS "+"*"*10)
        exp_generator = self.get_exp_generator()
        for exp in exp_generator:
            exp.run()
        self.logger.log("PROCESS COMPLETED!\n\n")


