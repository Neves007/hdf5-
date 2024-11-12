class Observer:
    """
    观察者基类
    定义了一个接口，用于接收更新通知。
    """
    def update(self, data):
        raise NotImplementedError("Subclasses should implement this!")