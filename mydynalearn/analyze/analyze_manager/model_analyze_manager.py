from mydynalearn.analyze.analyzer.model_analyzer import ModelAnalyzer
from .observer import Observer
class ModelAnalyzeManager(Observer):
    def __init__(self,config_analyze):
        self.config_analyze = config_analyze
        self.subscribers = []  # 存储所有的观察者（订阅者）  
        self.all_epoch_analyzer = []

    def subscribe(self, observer: Observer):
        """
        添加一个观察者到订阅者列表中。
        :param observer: 需要被通知的观察者对象
        """
        self.subscribers.append(observer)


    def notify(self, **data):
        """
        通知所有订阅者关于最佳 epoch 的分析结果。
        :param model_name: 当前模型的名称
        :param best_epoch_data: 当前模型的最佳 epoch 数据
        """
        for subscriber in self.subscribers:
            subscriber.update(**data)

    def update(self, **kwargs):
        """
        当收到 EpochAnalyzeManager 的更新时，存储并分析 epoch 数据。
        :param data: 包含模型名称和 epoch 数据的元组
        """
        process = kwargs['process']
        if process == "new epoch":
            epoch_analyzer = kwargs['epoch_analyzer']
            self.all_epoch_analyzer.append(epoch_analyzer)
        elif process == "new model":
            model_analyzer = ModelAnalyzer(self.config_analyze, self.all_epoch_analyzer)
            model_analyzer.run()
            self.notify(process=process, model_analyzer=model_analyzer)
            del self.all_epoch_analyzer
            self.all_epoch_analyzer = []
        elif process == "finish":
            self.notify(process=process)
