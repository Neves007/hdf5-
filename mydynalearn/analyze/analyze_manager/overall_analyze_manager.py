from mydynalearn.analyze.analyzer.overall_analyzer import OverallAnalyzer
from .observer import Observer

class OverallAnalyzeManager(Observer):
    """
    汇总所有模型的最佳 epoch 结果，找出整体上表现最优的模型。
    """

    def __init__(self, config_analyze):
        self.config_analyze = config_analyze
        self.all_model_analyzer = []


    def update(self, **kwargs):
        """
        当收到 ModelAnalyzeManager 的更新时，存储并分析每个模型的最佳结果。
        :param data: 包含模型名称和最佳 epoch 数据的元组
        """
        process = kwargs['process']
        if process == "new model":
            model_analyzer = kwargs['model_analyzer']
            self.all_model_analyzer.append(model_analyzer)
        elif process == "finish":
            overall_analyzer = OverallAnalyzer(self.config_analyze, self.all_model_analyzer)
            overall_analyzer.run()