from .analyze_result_handler import AnalyzeResultHandler
from mydynalearn.dynamics.utils import *
import torch


class EpochResultHandlerGeneralPerformance(AnalyzeResultHandler):
    def __init__(self, config_analyze, epoch_task):
        self.config_analyze = config_analyze
        self.epoch_task = epoch_task
        self.init_metadata()
        parent_group = self.get_parent_group()
        cur_group = self.epoch_task.cur_group
        super().__init__(parent_group, cur_group)

    def get_parent_group(self):
        epoch_analyze_dir = self.config_analyze['epoch_analyze_dir']
        parent_group = f"{epoch_analyze_dir}/GeneralPerformance"
        return parent_group

    def init_metadata(self):
        metadata = self.epoch_task.metadata
        self.set_metadata(metadata)

    def init_result(self):
        self.result = self.epoch_task.get_test_rulsult()
        self.convert_to_storage_format(self.result)

    def analyze_result(self):
        epoch_inde = self.result['epoch_index']
        mean_loss = self.result['mean_loss']
        node_loss = self.result['node_loss']
        R = self.result['R']
        STATES = self.metadata['STATES']
        x_lable = feature_to_lable(self.result['x'], STATES)
        y_pred_lable = feature_to_lable(self.result['y_pred'], STATES)
        y_true_lable = feature_to_lable(self.result['y_true'], STATES)
        y_ob_lable = feature_to_lable(self.result['y_ob'], STATES)
        trans_prob_true = extract_trans_prob(self.result['y_ob'], self.result['y_true'])
        trans_prob_pred = extract_trans_prob(self.result['y_ob'], self.result['y_pred'])
        true_trans_type = [get_transtype_latex(s, t) for s, t in zip(x_lable, y_ob_lable)]
        pred_trans_type = [get_transtype_latex(s, t) for s, t in zip(x_lable, y_pred_lable)]
        analyze_result = {
            "epoch_inde": epoch_inde,
            "mean_loss": mean_loss,
            "node_loss": node_loss,
            "R": R,
            "STATES": STATES,
            "x_lable": x_lable,
            "y_pred_lable": y_pred_lable,
            "y_true_lable": y_true_lable,
            "y_ob_lable": y_ob_lable,
            "trans_prob_true": trans_prob_true,
            "trans_prob_pred": trans_prob_pred,
            "true_trans_type": true_trans_type,
            "pred_trans_type": pred_trans_type,
        }
        return analyze_result
