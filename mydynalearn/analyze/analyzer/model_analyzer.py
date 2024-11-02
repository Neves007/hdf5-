from mydynalearn.dataset import *
from mydynalearn.analyze.utils.data_handler import *
from mydynalearn.config import ConfigFile
import os
import pickle
from mydynalearn.logger import Log
from mydynalearn.util.lazy_loader import PickleLazyLoader

class ModelAnalyzer():
    config = ConfigFile.get_config_analyze()['default']
    def __init__(self, config, exp, epoch_index):
        """
        初始化 ModelAnalyzer
        :param exp: 实验对象
        """
        self.exp = exp
        self.logger = Log("ModelAnalyzer")
        self.dynamics = self.exp.dynamics
        self.epoch_index = epoch_index

    def get_analyze_result_dir_path(self):
        """
        构建分析结果的文件路径
        """
        model_info = f"{self.exp.exp_info['model_network_name']}_{self.exp.exp_info['model_dynamics_name']}_{self.exp.exp_info['model_name']}"
        testdata_info = f"{self.exp.exp_info['dataset_network_name']}_{self.exp.exp_info['dataset_dynamics_name']}_{self.exp.exp_info['model_name']}"
        model_dir_name = f"model_{model_info}"
        testdata_dir_name = f"testdata_{testdata_info}"

        analyze_result_dir_path = os.path.join(
            self.config.root_dir_path,
            self.config.analyze_result_dir_name,
            model_dir_name,
            "normal_performance"
        )
        os.makedirs(analyze_result_dir_path, exist_ok=True)
        return analyze_result_dir_path

    def get_data_file(self, type='normal_performance'):
        analyze_result_dir_path = self.get_analyze_result_dir_path()
        analyze_result_filepath = os.path.join(analyze_result_dir_path,
                                               f"epoch{self.epoch_index}_{type}_analyze_result.pkl")
        return analyze_result_filepath

    def _init_test_model(self):
        test_model = self.exp.model.epoch_tasks
        test_model.load(self.epoch_index)
        return test_model

