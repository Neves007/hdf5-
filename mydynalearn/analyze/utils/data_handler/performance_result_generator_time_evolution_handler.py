import pandas as pd
from mydynalearn.analyze.utils.data_handler import DynamicDataHandlerTimeEvolution
class PerformanceResultGeneratorTimeEvolutionHandler():
    def __init__(self, exp, epoch_index, network, dynamics,dynamic_dataset_time_evolution):
        self.exp = exp
        self.epoch_index = epoch_index
        self.network = network
        self.dynamics = dynamics
        self.dynamic_dataset_time_evolution = dynamic_dataset_time_evolution

    def create_analyze_result(self):
        """
        创建分析结果
        """
        dynamic_data_handler_time_evolution = DynamicDataHandlerTimeEvolution(self.network, self.dynamics, self.dynamic_dataset_time_evolution)
        test_result_info = {
            "model_network_name": self.exp.exp_info["model_network_name"],
            "model_dynamics_name": self.exp.exp_info["model_dynamics_name"],
            "MAX_DIMENSION": self.dynamics.MAX_DIMENSION,
            "dataset_network_name": self.exp.exp_info["dataset_network_name"],
            "dataset_dynamics_name": self.exp.exp_info["dataset_dynamics_name"],
            "model_name": self.exp.exp_info["model_name"],
            "model_epoch_index": self.epoch_index,
        }
        test_result_df = dynamic_data_handler_time_evolution.get_evolution_result_dataframe(test_result_info)
        return {
            "test_result_info": test_result_info,
            "test_result_df": test_result_df,
            "network": self.network,
            "dynamics": self.dynamics,
            "dynamics_STATES_MAP": self.dynamics.STATES_MAP,
        }