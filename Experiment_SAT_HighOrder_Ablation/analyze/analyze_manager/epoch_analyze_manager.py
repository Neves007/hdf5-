from ...analyze.analyzer.epoch_analyzer import EpochAnalyzer
from .analyze_manager import AnalyzeManager


class EpochAnalyzeManager(AnalyzeManager):
    def __init__(self, exp_generator):
        """
        初始化 EpochAnalyzeManager 类，遍历实验和 `epoch_task`，创建并运行 `EpochAnalyzer` 实例，对 `epoch_task` 进行分析，并通过 `_notify` 方法向订阅者传递分析进度和结果。

        参数:
        - exp_generator: generator，实验生成器，用于提供实验对象（exp），每个实验包含多个 epoch_task。
        """
        self.exp_generator = exp_generator  # 实验生成器，用于生成不同的实验实例
        super().__init__()  # 调用父类的构造函数，初始化订阅者和输出处理器

    def run(self):
        """
        运行分析流程，对每个实验的所有 epoch_task 进行分析。

        对每个 epoch_task 创建 EpochAnalyzer 进行分析，分析完成后通知所有订阅者。
        """
        # 遍历所有的实验（exp）
        for exp in self.exp_generator():
            # 遍历实验中的所有 epoch_task
            for epoch_task in exp.model.epoch_task_generator():
                # 创建 EpochAnalyzer 实例并运行分析
                epoch_analyzer = EpochAnalyzer(epoch_task)
                epoch_analyzer.run()

                # 通知订阅者新的 epoch_analyzer 已创建，并传递其字典化的结果
                self._notify(process="new epoch_analyzer", epoch_analyzer_dict=epoch_analyzer.to_dict())

            # 通知订阅者当前实验的所有 epoch_task 分析已完成
            self._notify(process="new model")

        # 通知订阅者整个分析流程已完成
        self._notify(process="finish")
