from .analyze_result_handler import AnalyzeResultHandler
from mydynalearn.dynamics.utils import *
import torch

class ModelResultHandler(AnalyzeResultHandler):
    def __init__(self, config_analyze, all_epoch_analyzer):
        self.config_analyze = config_analyze
        self.all_epoch_analyzer = all_epoch_analyzer
        self.init_metadata()
        parent_group = self.get_parent_group()
        cur_group = (
                     f"NETWORK_NAME_{self.metadata['NETWORK_NAME']}_"
                     f"NUM_NODES_{self.metadata['NUM_NODES']}/"
                     f"DYNAMIC_NAME_{self.metadata['DYNAMIC_NAME']}_"
                     f"T_INIT_{self.metadata['T_INIT']}_"
                     f"SEED_FREC_{self.metadata['SEED_FREC']}/"
                     f"MODEL_NAME_{self.metadata['MODEL_NAME']}/"
                     )
        super().__init__(parent_group, cur_group)

    def get_parent_group(self):
        analyze_dir = self.config_analyze['model_analyze_dir']
        parent_group = f"{analyze_dir}"
        return parent_group

    def init_metadata(self):
        metadata = {}
        epoch_analyzer_metadata = self.all_epoch_analyzer[0].epoch_task.metadata
        metadata['DEVICE'] = epoch_analyzer_metadata['DEVICE']
        # network
        metadata['NETWORK_NAME'] = epoch_analyzer_metadata['NETWORK_NAME']
        metadata["NUM_NODES"] = epoch_analyzer_metadata['NUM_NODES']
        # dynamics
        metadata['DYNAMIC_NAME'] = epoch_analyzer_metadata['DYNAMIC_NAME']
        metadata["SEED_FREC"] = epoch_analyzer_metadata['SEED_FREC']
        metadata["STATES"] = epoch_analyzer_metadata['STATES']
        # dataset
        metadata['NUM_SAMPLES'] = epoch_analyzer_metadata['NUM_SAMPLES']
        metadata['T_INIT'] = epoch_analyzer_metadata['T_INIT']
        metadata['IS_WEIGHT'] = epoch_analyzer_metadata['IS_WEIGHT']
        # model
        metadata['EPOCHS'] = epoch_analyzer_metadata['EPOCHS']
        metadata['MODEL_NAME'] = epoch_analyzer_metadata['MODEL_NAME']
        self.metadata = metadata

    def init_result(self):
        self.all_epoch_analyze_result = [epoch_analyzer.get_analyze_result() for epoch_analyzer in self.all_epoch_analyzer]

    def analyze_result(self):
        # Find the dictionary with the maximum 'R' value
        best_epoch = max(self.all_epoch_analyze_result, key=lambda x: x['R'])
        return best_epoch
