from typing import List

class Observer:
    """
    观察者基类
    定义了一个接口，用于接收更新通知。
    """
    def update(self, data):
        raise NotImplementedError("Subclasses should implement this!")

class EpochAnalyzeManager:
    """
    负责每个模型中各个 epoch_analyzer 的结果分析，并通知观察者。
    """
    def __init__(self):
        self.subscribers = []  # 存储所有的观察者（订阅者）

    def subscribe(self, observer: Observer):
        """
        添加一个观察者到订阅者列表中。
        :param observer: 需要被通知的观察者对象
        """
        self.subscribers.append(observer)

    def notify(self, model_name, epoch_data):
        """
        通知所有订阅者分析结果。
        :param model_name: 当前模型的名称
        :param epoch_data: 当前 epoch_analyzer 的数据
        """
        for subscriber in self.subscribers:
            subscriber.update((model_name, epoch_data))

    def analyze_epoch(self, model_name, epoch_data):
        """
        分析每个模型的每个 epoch_analyzer 的数据。
        :param model_name: 当前模型的名称
        :param epoch_data: 当前 epoch_analyzer 的数据
        """
        # 这里模拟对每个 epoch_analyzer 数据的分析
        print(f"Analyzing epoch_analyzer {epoch_data['epoch_analyzer']} for model {model_name}: Accuracy={epoch_data['accuracy']}")
        # 分析完毕后通知订阅者
        self.notify(model_name, epoch_data)

class ModelAnalyzeManager(Observer):
    """
    汇总和分析每个模型的所有 epoch_analyzer 结果，找出最佳 epoch_analyzer，并分析时间演化趋势。
    """
    def __init__(self):
        self.model_best_epochs = {}  # 存储每个模型的最佳 epoch_analyzer 数据
        self.subscribers = []  # 存储所有的观察者（订阅者）
        self.model_epochs_data = {}  # 存储每个模型的所有 epoch_analyzer 数据

    def subscribe(self, observer: Observer):
        """
        添加一个观察者到订阅者列表中。
        :param observer: 需要被通知的观察者对象
        """
        self.subscribers.append(observer)

    def notify(self, model_name, best_epoch_data):
        """
        通知所有订阅者关于最佳 epoch_analyzer 的分析结果。
        :param model_name: 当前模型的名称
        :param best_epoch_data: 当前模型的最佳 epoch_analyzer 数据
        """
        for subscriber in self.subscribers:
            subscriber.update((model_name, best_epoch_data))

    def update(self, data):
        """
        当收到 EpochAnalyzeManager 的更新时，存储并分析 epoch_analyzer 数据。
        :param data: 包含模型名称和 epoch_analyzer 数据的元组
        """
        model_name, epoch_data = data
        # 存储每个模型的所有 epoch_analyzer 数据
        if model_name not in self.model_epochs_data:
            self.model_epochs_data[model_name] = []
        self.model_epochs_data[model_name].append(epoch_data)

        # 检查是否所有 epoch_analyzer 都已完成
        if len(self.model_epochs_data[model_name]) == self.epochs:
            # 分析所有 epoch_analyzer 数据，找出最佳 epoch_analyzer
            best_epoch_data = max(self.model_epochs_data[model_name], key=lambda x: x['accuracy'])
            self.model_best_epochs[model_name] = best_epoch_data
            print(f"Model {model_name} analysis complete. Best Epoch: {best_epoch_data['epoch_analyzer']}, Accuracy: {best_epoch_data['accuracy']}")
            # 通知订阅者当前模型的最佳 epoch_analyzer 数据
            self.notify(model_name, best_epoch_data)

class OverallAnalyzeManager(Observer):
    """
    汇总所有模型的最佳 epoch_analyzer 结果，找出整体上表现最优的模型。
    """
    def __init__(self):
        self.best_model = None  # 最优模型的名称
        self.best_epoch_data = None  # 最优模型的最佳 epoch_analyzer 数据
        self.models_best_results = {}  # 存储所有模型的最佳结果

    def update(self, data):
        """
        当收到 ModelAnalyzeManager 的更新时，存储并分析每个模型的最佳结果。
        :param data: 包含模型名称和最佳 epoch_analyzer 数据的元组
        """
        model_name, best_epoch_data = data
        # 存储每个模型的最佳 epoch_analyzer 结果
        self.models_best_results[model_name] = best_epoch_data

        # 检查是否所有模型都已完成分析
        if len(self.models_best_results) == len(self.models):
            # 找出整体上表现最优的模型
            for model, epoch_data in self.models_best_results.items():
                if self.best_epoch_data is None or epoch_data['accuracy'] > self.best_epoch_data['accuracy']:
                    self.best_model = model
                    self.best_epoch_data = epoch_data
            print(f"Updated overall best model: Model={self.best_model}, Epoch={self.best_epoch_data['epoch_analyzer']}, Accuracy={self.best_epoch_data['accuracy']}")

class AnalyzeManagerManager():
    def __init__(self, exp_generator):
        """
        初始化 AnalyzeManagerManager
        :param exp_generator: 提供实验生成器的管理对象
        """
        self.exp_generator = exp_generator  # 用于生成实验数据的对象
        self.epoch_analyze_manager = EpochAnalyzeManager()  # 初始化 Epoch 分析管理器
        self.model_analyze_manager = ModelAnalyzeManager()  # 初始化模型分析管理器
        self.overall_analyze_manager = OverallAnalyzeManager()  # 初始化总体分析管理器

        # 通过职责链将各个分析模块组装起来，使用观察者模式
        self.epoch_analyze_manager.subscribe(self.model_analyze_manager)
        self.model_analyze_manager.subscribe(self.overall_analyze_manager)

    def analyze(self, models: List[str], epochs: int):
        """
        对指定的模型和 epoch_analyzer 数量执行分析。
        :param models: 模型名称的列表
        :param epochs: 每个模型的 epoch_analyzer 数量
        """
        self.models = models  # 存储模型列表
        self.epochs = epochs  # 存储 epoch_analyzer 数量
        self.model_analyze_manager.epochs = epochs  # 设置模型分析管理器的 epoch_analyzer 数量
        self.overall_analyze_manager.models = models  # 设置总体分析管理器的模型列表

        # 对每个模型的每个 epoch_analyzer 进行分析
        for model in models:
            for epoch in range(epochs):
                # 生成当前模型和 epoch_analyzer 的数据
                epoch_data = self.exp_generator.generate_epoch_data(model, epoch)
                # 调用 EpochAnalyzeManager 进行分析
                self.epoch_analyze_manager.analyze_epoch(model, epoch_data)

# 示例代码，演示如何使用这些管理器
class DummyExpGenerator:
    """
    一个用于模拟实验数据生成的类
    """
    def generate_epoch_data(self, model_name, epoch):
        """
        生成模拟的 epoch_analyzer 数据。
        :param model_name: 模型名称
        :param epoch: 当前 epoch_analyzer 的编号
        :return: 模拟的 epoch_analyzer 数据，包括 epoch_analyzer 编号和 accuracy
        """
        return {"epoch_analyzer": epoch, "accuracy": 0.8 + 0.01 * epoch} # 假设每个 epoch_analyzer 的 accuracy 提高 0.01

if __name__ == "__main__":
    # 初始化实验数据生成器
    exp_generator = DummyExpGenerator()
    # 初始化分析管理器
    analyze_manager_manager = AnalyzeManagerManager(exp_generator)
    # 定义模型列表和 epoch_analyzer 数量
    models = ["GAT", "SAT"]
    epochs = 3

    # 执行分析
    analyze_manager_manager.analyze(models, epochs)
