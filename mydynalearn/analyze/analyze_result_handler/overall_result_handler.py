from .analyze_result_handler import AnalyzeResultHandler
from mydynalearn.dynamics.utils import *
import torch
import pandas as pd


class OverallResultHandler():
    def __init__(self, config_analyze, all_model_analyzer):
        self.config_analyze = config_analyze
        self.all_model_analyzer = all_model_analyzer


    def save(self, data):
        path = "output/overall.csv"
        data.to_csv(path, index=False)

    def run(self):
        self.all_model_analyze_result = [model_analyzer.get_analyze_result() for model_analyzer in
                                         self.all_model_analyzer]
        key_of_interest = ['R', 'mean_loss', 'NETWORK_NAME', 'NUM_NODES', 'DYNAMIC_NAME', 'SEED_FREC', 'T_INIT',
                           'MODEL_NAME', 'epoch_index']
        overall_data = {
            key: [model_analyze_result[key]
                  for model_analyze_result in self.all_model_analyze_result]
            for key in key_of_interest
        }

        overall_data = pd.DataFrame(overall_data)
        self.save(overall_data)
