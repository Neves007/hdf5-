# 获取配置
from mydynalearn.config.config_training_exp import ConfigTrainingExp
from .experiment_train import ExperimentTrain
from mydynalearn.logger import Log
from mydynalearn.util.params_dealer import PasramsDealer


class TrainExperimentManager():
    def __init__(self,fix_config_dict, train_params):
        self.fix_config_dict = fix_config_dict
        self.logger = Log("TrainExperimentManager")
        self.root_dir = r"./output/"
        self.train_params = train_params

    def fix_config(self,config,model_name):
        '''调整配置
        '''
        # T总时间步
        config.dataset.NUM_SAMPLES = self.fix_config_dict['NUM_SAMPLES']
        config.dataset.NUM_TEST = self.fix_config_dict['TESTSET_TIMESTEP']
        config.model.EPOCHS = self.fix_config_dict['EPOCHS']  # 10
        config.model.NAME = model_name
        NUM_STATES = len(config.dynamics.STATES_MAP.keys())
        config.model.in_channels[0] = NUM_STATES
        config.model.out_channels[-1] = NUM_STATES
        config.DEVICE = self.fix_config_dict['DEVICE']  # 10

    def get_loaded_model_exp(self, train_args, epoch_index):
        '''加载指定模型指定epoch_index的训练模型
        :param train_args: (network, dynamics, model, IS_WEIGHT)
        :param epoch_index: int
        :return: model_exp
        '''
        model_exp = self.get_train_exp(*train_args)
        model_exp.model.load_model(epoch_index)
        model_exp.create_dataset()
        return model_exp


    def get_train_exp(self,network, dynamics, model, IS_WEIGHT=False):
        '''通过参数获得实验对象

        :param network:
        :param dynamics:
        :param model:
        :param IS_WEIGHT:
        :return: exp
        '''
        exp_name = "dynamicLearning-" + network + "-" + dynamics + "-" + model
        kwargs = {
            "NAME": exp_name,
            "network": network,
            "dynamics": dynamics,
            "IS_WEIGHT": IS_WEIGHT,
            "seed": 0,
            "root_dir": self.root_dir,
        }
        config = ConfigTrainingExp(**kwargs)
        self.fix_config(config, model)
        exp = ExperimentTrain(config)
        return exp

    def get_exp_generator(self):
        for train_param in self.train_params:
            exp = self.get_train_exp(*train_param)
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


