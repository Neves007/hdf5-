import pandas as pd
from .draw_handler import DrawHadler
import os
from Experiment_SAT_HighOrder_LossK1K2.analyze.analyzer.epoch_analyzer import EpochAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from mydynalearn.config import *
from mydynalearn.logger import Log
from matplotlib.colors import Normalize
import numpy as np


class LossK1K2Handler(DrawHadler):
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
        self.axis_lable_fs = 12  # 坐标轴标签字体大小
        self.legend_fs = 12  # 图例字体大小
        super().__init__(visualizer_dir, visualizer_data)

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
        '''
        # 设置标题，包含相关系数 R 的值
        self.ax.set_title(r'$R$ = {:0.5f}'.format(self.corrcoef), fontsize=self.title_fs)
        # 去除左边和底部的边框



    def _draw(self):
        '''
        绘制图像，散点透明度为0.1，图例为高透明度
        '''
        sns.set_theme(style="whitegrid")
        # 创建绘图区域
        keys_to_extract = ["NUM_NEIGHBOR_NODES",
                           "NUM_NEIGHBOR_TRIANGLES",
                           "node_mse_loss",
                           "node_mse_loss"]

        # 提取包含 keys_to_extract 键的子集
        subset = self.extract_to_dataframe(keys_to_extract)
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        avg_subset = subset.groupby(['NUM_NEIGHBOR_NODES', 'NUM_NEIGHBOR_TRIANGLES']).mean().reset_index()

        # 使用 Seaborn 绘图
        g = sns.scatterplot(
            data=avg_subset,
            x="NUM_NEIGHBOR_NODES",
            y="NUM_NEIGHBOR_TRIANGLES",
            hue="node_mse_loss",  # 节点颜色基于平均 node_mse_loss
            size="node_mse_loss",  # 节点大小基于平均 node_mse_loss
            palette=self.palette,
            hue_norm=Normalize(vmin=0.0001, vmax=0.0010),
            edgecolor=".7",
            sizes=(50, 300),  # 调整节点大小范围
            size_norm=(0.0001, 0.0010),  # 正规化大小范围
            ax=self.ax  # 使用指定的 ax
        )

        # 设置 x 和 y 轴标签，并将图表的长宽比设置为相等
        self.ax.set_xlabel("NUM_NEIGHBOR_NODES")
        self.ax.set_ylabel("NUM_NEIGHBOR_TRIANGLES")
        self.ax.set_aspect("equal")

        # 使用 sns.despine 去除左边和底部边框
        sns.despine(ax=self.ax, left=True, bottom=True)

        # 设置 xticks 和 yticks 从最小值到最大值，间隔为 1
        x_min, x_max = avg_subset["NUM_NEIGHBOR_NODES"].min(), avg_subset["NUM_NEIGHBOR_NODES"].max()
        y_min, y_max = avg_subset["NUM_NEIGHBOR_TRIANGLES"].min(), avg_subset["NUM_NEIGHBOR_TRIANGLES"].max()
        self.ax.set_xticks(np.arange(x_min, x_max + 1, 1))
        self.ax.set_yticks(np.arange(y_min, y_max + 1, 1))
        self.ax.tick_params(axis='x', rotation=90)  # 设置标签角度和对齐

        # 自定义图例
        norm = Normalize(vmin=0.0001, vmax=0.0010)
        legend_labels = ['Low', 'Medium-Low','Medium', 'Medium-High', 'High']
        num_legend = len(legend_labels)
        legend_sizes = np.linspace(50, 300, num_legend)
        legend_colors = plt.cm.get_cmap(self.palette)(
            norm(np.linspace(0.0001, 0.0010, num_legend)))

        handles = [plt.scatter([], [], s=size, color=color, edgecolor='.7') for size, color in
                   zip(legend_sizes, legend_colors)]

        self.ax.legend(handles, legend_labels, title="node mse loss", fontsize=self.legend_fs,
                       title_fontsize=self.legend_fs, loc='upper left', bbox_to_anchor=(1.01, 1))
        plt.tight_layout()
