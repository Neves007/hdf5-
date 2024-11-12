from mydynalearn.analyze.analyzer.epoch_analyzer import EpochAnalyzer
from .observer import Observer

class EpochAnalyzeManager():
    def __init__(self,config_analyze,exp_generator):
        self.config_analyze = config_analyze
        self.exp_generator = exp_generator
        self.subscribers = []  # 存储所有的观察者（订阅者）

    def subscribe(self, observer: Observer):
        """
        添加一个观察者到订阅者列表中。
        :param observer: 需要被通知的观察者对象
        """
        self.subscribers.append(observer)

    def notify(self, *args,**kwargs):
        """
        通知所有订阅者分析结果。
        :param model_name: 当前模型的名称
        :param epoch_data: 当前 epoch 的数据
        """
        for subscriber in self.subscribers:
            subscriber.update(**kwargs)

    def epoch_analyzer_generator(self):
        '''当exp为None时，遍历所有exp进行分析'''
        for exp in self.exp_generator():
            for epoch_task in exp.model.epoch_task_generator():
                yield EpochAnalyzer(self.config_analyze, epoch_task)

    def run(self):
        '''
        EpochAnalyzer 对每个epoch_task结果进行分析
        :return:
        '''
        for exp in self.exp_generator():
            for epoch_task in exp.model.epoch_task_generator():
                epoch_analyzer = EpochAnalyzer(self.config_analyze, epoch_task)
                epoch_analyzer.run()
                self.notify(process="new epoch",epoch_analyzer=epoch_analyzer)
            self.notify(process="new model")
        self.notify(process="finish")
