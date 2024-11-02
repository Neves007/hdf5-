from mydynalearn.dataset import *
from mydynalearn.analyze.utils.data_handler import *
from mydynalearn.config import ConfigFile

from .model_analyzer import ModelAnalyzer
from mydynalearn.util.lazy_loader import PickleLazyLoader


class ModelAnalyzerTimeEvolution(PickleLazyLoader, ModelAnalyzer):
    config = ConfigFile.get_config_analyze()['default']
    def __init__(self, config, exp, epoch_index):
        """
        初始化 ModelAnalyzer
        :param exp: 实验对象
        """
        ModelAnalyzer.__init__(self,config, exp, epoch_index)
        self.data_file = self.get_data_file(type='time_evolution')
        PickleLazyLoader.__init__(self,self.data_file)

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
            "time_evolution"
        )
        os.makedirs(analyze_result_dir_path, exist_ok=True)
        return analyze_result_dir_path

    def create_dynamic_dataset_time_evolution(self):
        """
        创建时间演化的动力学数据集
        """
        self.dynamics_dataset_time_evolution = DynamicDatasetTimeEvolution(self.exp, self.test_model, self.network,
                                                                           self.dynamics)
        return self.dynamics_dataset_time_evolution.get_data()


    def analyze_model_performance_time_evolution(self):
        """分析时间演化数据
        :return:
        """
        self.logger.increase_indent()
        self.logger.log(f"analyze time evolution model performance of epoch {self.epoch_index}")
        dynamic_dataset_time_evolution = self.create_dynamic_dataset_time_evolution()
        handler_performance_generator_time_evolution = PerformanceResultGeneratorTimeEvolutionHandler(self.exp,
                                                                                                      self.epoch_index,
                                                                                                      self.network,
                                                                                                      self.dynamics,
                                                                                                      dynamic_dataset_time_evolution)
        analyze_result_model_performance_time_evolution = handler_performance_generator_time_evolution.create_analyze_result()
        self.logger.decrease_indent()
        return analyze_result_model_performance_time_evolution

    def _create_data(self):
        self.network = self.exp.network.get_data()
        self.test_model = self._init_test_model()
        analyze_result = self.analyze_model_performance_time_evolution()
        return analyze_result

    @staticmethod
    def load_from_dataframe_item(dataframe_item):
        type = "time_evolution"
        config = ModelAnalyzer.config
        epoch_index = dataframe_item['model_epoch_index']
        model_info = f"{dataframe_item['model_network_name']}_{dataframe_item['model_dynamics_name']}_{dataframe_item['model_name']}"
        testdata_info = f"{dataframe_item['dataset_network_name']}_{dataframe_item['dataset_dynamics_name']}_{dataframe_item['model_name']}"
        model_dir_name = f"model_{model_info}"
        testdata_dir_name = f"testdata_{testdata_info}"

        analyze_result_dir_path = os.path.join(
            config.root_dir_path,
            config.analyze_result_dir_name,
            model_dir_name,
            type
        )
        analyze_result_filepath = os.path.join(analyze_result_dir_path,
                                               f"epoch{epoch_index}_{type}_analyze_result.pkl")
        if not os.path.exists(analyze_result_filepath):
            raise FileNotFoundError(f"No such file: {analyze_result_filepath}")
        with open(analyze_result_filepath, "rb") as f:
            result = pickle.load(f)
        return result