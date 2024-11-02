from mydynalearn.dataset import *
from mydynalearn.analyze.utils.data_handler import *
from mydynalearn.config import ConfigFile
import os
import pickle
from mydynalearn.logger import Log
from mydynalearn.util.lazy_loader import PickleLazyLoader
from .model_analyzer import ModelAnalyzer

class ModelAnalyzerNormalPerformance(PickleLazyLoader, ModelAnalyzer):
    def __init__(self, config, exp, epoch_index):
        """
        初始化 ModelAnalyzer
        :param exp: 实验对象
        """
        ModelAnalyzer.__init__(self,config, exp, epoch_index)
        self.data_file = self.get_data_file(type='normal_performance')
        PickleLazyLoader.__init__(self,self.data_file)

    def analyze_model_performance(self):
        """
        :return:
        """
        dataset = self.exp.dataset.load()
        self.test_model = self._init_test_model()
        generator_performance_result = self.test_model.run_test_epoch(**dataset)
        handler_performance_generator = PerformanceResultGeneratorHandler(self.exp,
                                                                          self.epoch_index,
                                                                          generator_performance_result,**dataset)
        analyze_result = handler_performance_generator.create_analyze_result()
        return analyze_result

    def _create_data(self):
        self.network = self.exp.network.get_data()
        analyze_result = self.analyze_model_performance()
        return analyze_result


    @staticmethod
    def load_from_dataframe_item(dataframe_item):
        type = "normal_performance"
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