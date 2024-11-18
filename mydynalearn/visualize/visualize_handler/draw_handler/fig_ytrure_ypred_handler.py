from .draw_handler import DrawHadler
import os
from Experiment_SAT_HighOrder_LossK1K2.analyze.analyzer.epoch_analyzer import EpochAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from mydynalearn.config import *
from mydynalearn.logger import Log


class FigYtrureYpredHandler(DrawHadler):
    def __init__(self, visualizer_dir, visualizer_data):
        self.epoch_analyzer = EpochAnalyzer.from_dict(**visualizer_data)
        self.dataset = self.epoch_analyzer.epoch_general_performance_handler.get_dataset()
        palette_list = {
            "UAU": 'viridis',
            "CompUAU": 'viridis',
            "CoopUAU": 'viridis',
            "AsymUAU": 'viridis',
            "SCUAU": 'plasma',
            "SCCompUAU": 'plasma',
            "SCCoopUAU": 'plasma',
            "SCAsymUAU": 'plasma',
        }
        self.palette = palette_list[visualizer_data['DYNAMIC_NAME']]
        self.epoch_index = visualizer_data['epoch_index']
        self.corrcoef = visualizer_data['R']
        self.title_fs = 16  # 标题字体大小
        self.axis_lable_fs = 14  # 坐标轴标签字体大小
        self.legend_fs = 12  # 图例字体大小
        super().__init__(visualizer_dir, visualizer_data)
        pass

    def _set_fig_path(self):
        fig_dir = os.path.join(self.visualizer_dir, (
            f"MODEL_NAME_{self.visualizer_data['MODEL_NAME']}/"
            f"NETWORK_NAME_{self.visualizer_data['NETWORK_NAME']}/"
            f"DYNAMIC_NAME_{self.visualizer_data['DYNAMIC_NAME']}"))
        fig_name = (
            f"{self.visualizer_data['MODEL_NAME']}-"
            f"{self.visualizer_data['NETWORK_NAME']}-"
            f"{self.visualizer_data['DYNAMIC_NAME']}-"
            f"SEED_FREC_{self.visualizer_data['SEED_FREC']}-"
            f"NUM_SAMPLES_{self.visualizer_data['NUM_SAMPLES']}-"
            f"T_INIT_{self.visualizer_data['T_INIT']}-"
            f"epoch_index_{self.visualizer_data['epoch_index']}.png")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        self.fig_path = os.path.join(fig_dir, fig_name)

    def _edit_ax(self):
        '''
        编辑ax
        :return:
        '''
        self.ax.set_title(r' $R$ = {:0.5f}'.format(self.corrcoef), fontsize=self.title_fs)
        # self.ax.set_title(r'epoch_analyzer = {:d}, $R$ = {:0.5f}'.format(self.epoch_index, self.corrcoef))
        self.ax.set_xticks(np.linspace(0, 1, 5))
        self.ax.set_yticks(np.linspace(0, 1, 5))
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        self.ax.set_xlabel("Target", fontsize=self.axis_lable_fs)  # 设置x轴标注
        self.ax.set_ylabel("Prediction", fontsize=self.axis_lable_fs)  # 设置y轴标注
        # 图例已在 draw 方法中手动添加，这里不需要再次添加
        self.ax.grid(True)

    def _draw(self):
        '''
        绘制图像，散点透明度为0.1，图例为高透明度
        '''
        # 创建绘图区域
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        # 绘制散点图，设置透明度为0.1
        scatter = sns.scatterplot(
            x="trans_prob_true", y="trans_prob_pred",
            hue="pred_trans_type",
            palette=self.palette,
            s=70,  # 点的大小
            linewidth=0,
            alpha=0.1,  # 设置散点的透明度为0.1
            data=self.dataset, ax=self.ax
        )

        # 获取唯一的 `pred_trans_type`
        unique_types = np.unique(self.dataset['pred_trans_type'])

        # 获取调色板对应的颜色
        num_colors = len(unique_types)
        palette = sns.color_palette(self.palette, n_colors=num_colors)
        color_mapping = dict(zip(unique_types, palette))

        # 手动创建自定义图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=unique_type,
                   markerfacecolor=color_mapping[unique_type], markersize=10, alpha=0.8)
            for unique_type in unique_types
        ]

        # 添加自定义图例到图形
        self.ax.legend(handles=legend_elements, title="Transition type", loc='upper left', fontsize=self.legend_fs)
