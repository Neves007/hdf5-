from ...analyze.analyze_result_handler.epoch_analyzer.epoch_general_performance_handler import EpochGeneralPerformanceHandler
from mydynalearn.config import ConfigExp
from mydynalearn.model.epoch_tasks import EpochTasks

class EpochAnalyzer():
    def __init__(self, epoch_task):
        """
        初始化 EpochAnalyzer 实例。管理和执行 `epoch` 级别的性能分析，具体包括初始化分析组件、运行分析任务、提取和格式化分析结果，以及提供从字典恢复实例的功能。

        参数:
        - epoch_task: EpochTasks 对象，表示当前 epoch 的任务信息，包括元数据和数据集。
        """
        self.epoch_task = epoch_task
        # 初始化 EpochGeneralPerformanceHandler 以处理通用性能分析
        self.epoch_general_performance_handler = EpochGeneralPerformanceHandler(epoch_task)

    def run(self):
        """
        运行通用性能分析。调用 EpochGeneralPerformanceHandler 的 run 方法来执行分析。
        """
        self.epoch_general_performance_handler.run()

    def to_dict(self):
        """
        将当前 EpochAnalyzer 实例转换为字典格式，包含关键信息和元数据。

        返回:
        - dict: 包含分析配置和数据集的关键信息的字典。
        """
        metadata = self.epoch_general_performance_handler.get_metadata()  # 获取元数据
        # 构造字典，包含必要的元数据和数据集信息
        dict = {
            "NETWORK_NAME": metadata['NETWORK_NAME'],
            "DYNAMIC_NAME": metadata['DYNAMIC_NAME'],
            "MODEL_NAME": metadata['MODEL_NAME'],
            "IS_WEIGHT": metadata['IS_WEIGHT'],
            "DEVICE": metadata['DEVICE'],
            "NUM_NODES": metadata['NUM_NODES'],
            "SEED_FREC": metadata['SEED_FREC'],
            "NUM_SAMPLES": metadata['NUM_SAMPLES'],
            "T_INIT": metadata['T_INIT'],
            "EPOCHS": metadata['EPOCHS'],
            "epoch_index": metadata['epoch_index'],
            "R": metadata['R'],
            "mean_loss": metadata['mean_loss'],
        }
        return dict

    @classmethod
    def from_dict(cls, **info_dict):
        """
        从字典创建 EpochAnalyzer 实例。

        参数:
        - config_analyze: dict，分析配置文件，用于设置分析参数。
        - info_dict: dict，包含实例化 EpochAnalyzer 所需的所有信息。

        返回:
        - EpochAnalyzer 实例
        """
        # 创建 ConfigExp 实例，提供实验的基本配置
        exp_config = ConfigExp(**info_dict)
        # 创建 EpochTasks 实例，提供当前 epoch 的任务信息
        epoch_task = EpochTasks(exp_config, **info_dict)
        # 返回创建的 EpochAnalyzer 实例
        return cls(
            epoch_task=epoch_task,
        )
