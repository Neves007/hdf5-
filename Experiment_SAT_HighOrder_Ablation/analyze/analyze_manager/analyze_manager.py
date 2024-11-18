class AnalyzeManager():
    def __init__(self):
        """
        初始化 AnalyzeManager 类，创建空的订阅者列表和绘图器列表。
        """
        self.subscribers = []  # 用于存储所有观察者（订阅者）对象
        self.output_handlers = {}  # 用于输出处理：绘图，绘表

    def subscribe(self, observer):
        """
        将一个观察者添加到订阅者列表中，便于后续通知。

        :param observer: 需要被通知的观察者对象，必须包含 `update` 方法。
        """
        self.subscribers.append(observer)

    def _notify(self, **kwargs):
        """
        通知所有订阅者传递的分析结果或信息。

        :param kwargs: 包含通知内容的字典数据，传递给每个订阅者的 `update` 方法。
        """
        for subscriber in self.subscribers:
            subscriber.update(**kwargs)



    def update(self, **kwargs):
        """
        该方法可以由子类实现，用于接收外部更新信息。

        :param kwargs: 更新信息的字典数据，具体内容由实现决定。
        """
        pass
