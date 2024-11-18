from ....analyze.analyze_result_handler.analyze_result_handler import \
    AnalyzeResultHandler
from mydynalearn.dynamics.utils import *
from mydynalearn.config import ConfigFile


class EpochGeneralPerformanceHandler(AnalyzeResultHandler):
    def __init__(self, epoch_task):
        """
        初始化 EpochGeneralPerformanceHandler 实例。从 `epoch` 任务中提取和分析性能数据，将结果转换为便于后续使用的格式，并为每个 `epoch` 生成详细的性能报告。

        参数:
        - epoch_task: object，当前 epoch 的任务对象，包含元数据和测试结果。
        """
        self.config_analyze = ConfigFile.get_config_analyze()
        self.epoch_task = epoch_task
        self.init_metadata()  # 初始化元数据
        parent_group = self.get_parent_group()  # 获取父组路径
        cur_group = self.epoch_task.cur_group  # 当前组名称
        super().__init__(parent_group, cur_group)  # 调用父类构造函数

    def get_parent_group(self):
        """
        获取父组的路径。

        返回:
        - str: 用于存储通用性能分析结果的父组路径。
        """
        analyze_dir = self.config_analyze['epoch_analyze_dir']  # 从配置中获取分析目录
        parent_group = f"{analyze_dir}/GeneralPerformance"  # 拼接父组路径
        return parent_group

    def init_metadata(self):
        """
        初始化元数据，将 epoch_task 的元数据设置到当前对象。
        """
        metadata = self.epoch_task.metadata  # 获取任务的元数据
        self.set_metadata(metadata)  # 设置元数据

    def init_result(self):
        """
        初始化结果，将 epoch_task 的测试结果转换为存储格式。
        """
        self.result = self.epoch_task.get_test_result()  # 获取测试结果
        self.convert_to_storage_format(self.result)  # 转换结果为存储格式

    def analyze_result(self):
        """
        分析结果，提取并格式化分析信息。

        返回:
        - dict: 包含各种分析结果的字典。
        """
        epoch_index = self.result['epoch_index']  # 当前 epoch 的索引
        mean_loss = self.result['mean_loss']  # 平均损失
        node_mse_loss = self.result['node_mse_loss']  # 节点损失
        R = self.result['R']  # 其他相关指标
        STATES = self.metadata['STATES']  # 获取状态元数据

        # 将 x, y_pred, y_true, y_ob 特征转换为标签
        x_lable = feature_to_lable(self.result['x'], STATES)
        y_pred_lable = feature_to_lable(self.result['y_pred'], STATES)
        y_true_lable = feature_to_lable(self.result['y_true'], STATES)
        y_ob_lable = feature_to_lable(self.result['y_ob'], STATES)

        # 提取真实和预测的转移概率
        trans_prob_true = extract_trans_prob(self.result['y_ob'], self.result['y_true'])
        trans_prob_pred = extract_trans_prob(self.result['y_ob'], self.result['y_pred'])

        # 获取真实和预测的转移类型（LaTeX 格式）
        true_trans_type = [get_transtype_latex(s, t) for s, t in zip(x_lable, y_ob_lable)]
        pred_trans_type = [get_transtype_latex(s, t) for s, t in zip(x_lable, y_pred_lable)]

        # 将所有分析结果汇总到字典中
        analyze_result = {
            "epoch_index": epoch_index,
            "mean_loss": mean_loss,
            "node_mse_loss": node_mse_loss,
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
