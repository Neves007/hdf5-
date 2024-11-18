from ..analyze.analyze_manager import *
from mydynalearn.config.yaml_config.configfile import ConfigFile

class Analyze():
    def __init__(self, exp_generator):
        """
        初始化 Analyze 类，负责初始化分析流程，协调 `epoch` 分析和结果聚合，并确保分析结果在实验和 `epoch` 之间自动汇总，从而简化和集中化整个分析管理流程。
        参数:
        - exp_generator: generator，实验生成器，用于提供实验对象（exp），每个实验包含多个 epoch_task。
        """
        # 获取分析配置文件，用于初始化分析管理器
        self.exp_generator = exp_generator  # 提供实验对象的生成器

        # 初始化 Epoch 分析管理器，用于逐个 epoch 的分析
        self.epoch_analyze_manager = EpochAnalyzeManager(exp_generator)

        # 初始化结果聚合器，用于汇总所有 epoch 的分析结果
        self.results_aggregator_analyze_manager = ResultsAggregatorAnalyzeManager()

        # 将结果聚合器订阅到 epoch 分析管理器，以便在分析过程中接收每个 epoch 的分析结果
        self.epoch_analyze_manager.subscribe(self.results_aggregator_analyze_manager)

    def run(self):
        """
        执行分析过程，对所有实验的每个 epoch 进行分析。
        """
        # 运行 epoch 分析管理器，依次分析每个实验的所有 epoch，并将结果传递给订阅的聚合器
        self.epoch_analyze_manager.run()

    def get_output(self):
        return self.results_aggregator_analyze_manager.get_output()