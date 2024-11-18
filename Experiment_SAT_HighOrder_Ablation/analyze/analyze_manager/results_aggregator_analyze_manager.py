from .analyze_manager import AnalyzeManager
from ..analyzer.results_aggregator_analyzer import ResultsAggregatorAnalyzer
class ResultsAggregatorAnalyzeManager(AnalyzeManager):
    def __init__(self):
        super().__init__()
        self.results_aggregator_analyzer = ResultsAggregatorAnalyzer()
        self.all_epoch_analyzer_dict = []

    def update(self, **kwargs):
        """
        当收到 EpochAnalyzeManager 的更新时，存储并分析 epoch_analyzer 数据。
        :param data: 包含模型名称和 epoch_analyzer 数据的元组
        """
        process = kwargs['process']
        if process == "new epoch_analyzer":
            epoch_analyzer_dict = kwargs['epoch_analyzer_dict']
            self.all_epoch_analyzer_dict.append(epoch_analyzer_dict)
        elif process == "finish":
            self.results_aggregator_analyzer.run(self.all_epoch_analyzer_dict)

    def get_output(self):
        return self.results_aggregator_analyzer.get_output()