import matplotlib.pyplot as plt
import pandas as pd
class DrawHadler:
    """
    Drawer 类是一个基础绘图器类，提供了绘图、保存以及路径设置的框架方法。
    具体的绘图逻辑和路径设置由子类实现。
    """

    def __init__(self, visualizer_dir, visualizer_data):
        """
        初始化 Drawer 类的实例，设置默认的输出目录和输出路径。
        """
        self.visualizer_dir = visualizer_dir
        self.visualizer_data = visualizer_data
        self._set_fig_path()

    def extract_to_dataframe(self, keys_to_extract):
        subset = pd.DataFrame({k: self.dataset[k] for k in keys_to_extract if k in self.dataset})
        return subset

    def _save(self):
        """
        保存绘制的图像到指定路径，并关闭当前图像，释放内存。

        :raises AttributeError: 如果未创建 `self.fig`，则保存时会报错。
        """
        self.fig.savefig(self.fig_path)  # 将图像保存到指定路径
        plt.close(self.fig)  # 关闭图像，避免内存泄露


    def _draw(self):
        """
        绘制图像的抽象方法，需在子类中实现具体的绘图逻辑。

        :return: 生成的图像对象。
        :raises NotImplementedError: 如果子类未实现此方法，则抛出异常。
        """
        raise NotImplementedError("Subclasses should implement this!")

    def _set_fig_path(self):
        """
        设置图像输出路径的抽象方法，需在子类中实现具体的路径生成逻辑。

        :return: 图像的保存路径。
        :raises NotImplementedError: 如果子类未实现此方法，则抛出异常。
        """
        raise NotImplementedError("Subclasses should implement this!")

    def run(self):
        """
        执行绘图流程：调用 `draw` 方法生成图像，并调用 `save` 方法保存图像。

        :raises NotImplementedError: 如果子类未实现 `draw` 方法，则抛出异常。
        """
        self._draw()  # 调用子类实现的绘图方法
        self._edit_ax()  # 调用子类实现的绘图方法
        self._save()  # 保存图像
