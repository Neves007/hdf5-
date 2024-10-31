import os.path

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
from mydynalearn.drawer.utils.utils import _get_metrics
import seaborn as sns
from mydynalearn.config import *
import pandas as pd
import re
from mydynalearn.logger import Log
import itertools


class MatplotDrawer():
    def __init__(self):
        # self.shrink_times = 1
        self.task_name = 'MatplotDrawer'
        self.logger = Log("MatplotDrawer")
        self.title_fs = 16
        self.axis_lable_fs = 14
        self.legend_fs = 13
        self.config_drawer = Config.get_config_drawer()
        pass

    def save_fig(self):
        self.logger.increase_indent()
        fig_file_path = self.get_fig_file_path()
        self.fig.savefig(fig_file_path)
        self.logger.log("saved in:  " + fig_file_path)
        plt.close(self.fig)
        self.logger.decrease_indent()

    def _make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def get_fig_file_path(self, *args, **kwargs):
        '''
        获取图片的储存路径
        :param data_info: 数据的基本信息
        :return: fig_file_path
        '''
        # 初始化数据
        network_name = self.test_result_info["model_network_name"]
        dynamics_name = self.test_result_info["model_dynamics_name"]
        model_name = self.test_result_info["model_name"]
        epoch_index = self.test_result_info['model_epoch_index']
        # 获得模型整体信息
        model_info = str.join('_', [network_name, dynamics_name, model_name])
        # 图片名称
        fig_name = "{}_{}_epoch_{}".format(self.task_name, model_info, epoch_index)
        '_'.join([self.task_name, model_info, 'epoch', str(epoch_index)])
        # 图片所在目录
        fig_dir_root_path = self.config.fig_dir_path
        fig_dir_path = os.path.join(fig_dir_root_path, network_name)
        self._make_dir(fig_dir_path)
        fig_file_path = os.path.join(fig_dir_path, fig_name)
        return fig_file_path


class FigYtrureYpred(MatplotDrawer):
    def __init__(self,
                 test_result_info,
                 test_result_df,
                 dynamics_STATES_MAP,
                 model_performance_dict, *args, **kwargs
                 ):
        super().__init__()
        self.task_name = 'FigYtrureYpred'
        self.config = self.config_drawer['fig_ytrure_ypred']
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
        self.test_result_info = test_result_info
        self.palette = palette_list[test_result_info['model_dynamics_name']]
        self.epoch_index = test_result_info['model_epoch_index']
        self.corrcoef = model_performance_dict['R']
        self.STATES_MAP = dynamics_STATES_MAP
        self.test_result_df = test_result_df
        self.title_fs = 16  # 标题字体大小
        self.axis_lable_fs = 14  # 坐标轴标签字体大小
        self.legend_fs = 12  # 图例字体大小

    def edit_ax(self):
        '''
        编辑ax
        :return:
        '''
        self.ax.set_title(r' $R$ = {:0.5f}'.format(self.corrcoef), fontsize=self.title_fs)
        # self.ax.set_title(r'epoch = {:d}, $R$ = {:0.5f}'.format(self.epoch_index, self.corrcoef))
        self.ax.set_xticks(np.linspace(0, 1, 5))
        self.ax.set_yticks(np.linspace(0, 1, 5))
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        self.ax.set_xlabel("Target", fontsize=self.axis_lable_fs)  # 设置x轴标注
        self.ax.set_ylabel("Prediction", fontsize=self.axis_lable_fs)  # 设置y轴标注
        # 图例已在 draw 方法中手动添加，这里不需要再次添加
        self.ax.grid(True)

    def draw(self):
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
            data=self.test_result_df, ax=self.ax
        )

        # 获取唯一的 `pred_trans_type`
        unique_types = self.test_result_df['pred_trans_type'].unique()

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

class FigBetaRho():
    def __init__(self, dynamics, x, stady_rho_dict, **kwargs):
        self.task_name = 'FigBetaRho'
        self.config = self.config_drawer['fig_beta_rho']
        self.x = x
        self.stady_rho_dict = stady_rho_dict
        self.state_list = dynamics.STATES_MAP.keys()
        self.dynamic_name = dynamics.NAME
        self.num_fig = len(self.state_list)

    def draw(self):
        self.fig, ax = plt.subplots(1, self.num_fig, figsize=(5 * self.num_fig, 4.5))
        for i, state in enumerate(self.state_list):
            y = self.stady_rho_dict[state]
            ax[i].set_title('{} dynamic model'.format(self.dynamic_name))
            ax[i].plot(self.x, y, marker='^')  # Plot x vs. y with circle markers
            ax[i].set_xlabel("Effective Infection Rate")  # X-axis label
            ax[i].set_ylabel("$\\rho_{{{:s}}}$".format(state))  # Y-axis label
            ax[i].set_ylim([0, 1])  # Set the limits for the y-axis
            ax[i].set_xlim([0, self.x[-1]])  # Set the limits for the y-axis
            ax[i].grid(True)  # Show grid
        plt.tight_layout()


class FigConfusionMatrix(MatplotDrawer):
    def __init__(self,
                 test_result_info,
                 test_result_df,
                 dynamics_STATES_MAP,
                 model_performance_dict, *args, **kwargs):
        super(FigConfusionMatrix, self).__init__()
        self.test_result_info = test_result_info
        self.task_name = 'FigConfusionMatrix'
        self.config = self.config_drawer['fig_confusion_matrix']
        self.confusion_matrix = model_performance_dict['cm']
        self.STATES_MAP = dynamics_STATES_MAP

    def draw(self):
        # Draw a heatmap with the numeric values in each cell
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.subplots_adjust(top=0.981, bottom=0.123, left=0.091, right=0.99, hspace=0.275, wspace=0.375)
        sns.heatmap(self.confusion_matrix, annot=True, fmt=".2%", linewidths=.5, ax=ax, cmap="Blues")
        self.ax = ax
        self.fig = fig

    def edit_ax(self):
        self.ax.set_title("Normalized Confusion Matrix")
        self.ax.set_ylabel("Predicted lable")  # 设置x轴标注
        self.ax.set_xlabel("True lable")  # 设置y轴标注
        self.ax.grid(False)
        plt.tight_layout()


class FigActiveNeighborsTransprob(MatplotDrawer):
    def __init__(self,
                 test_result_info,
                 test_result_df,
                 dynamics_STATES_MAP,
                 model_performance_dict, *args, **kwargs):
        super().__init__()
        self.task_name = 'FigActiveNeighborsTransprob'
        self.config = self.config_drawer['fig_active_neighbors_transprob']
        self.test_result_info = test_result_info
        self.STATES_MAP = dynamics_STATES_MAP
        self.test_result_df = test_result_df
        self.shrink_times = 1.5

    def edit_ax(self):
        '''
        编辑ax
        :return:
        '''
        # self.ax.set_title(r' $R$ = {:0.5f}'.format(self.corrcoef))
        # self.ax.set_xticks(np.linspace(0,1,5))
        # 自定义图例
        from matplotlib.lines import Line2D
        # 获取唯一的 transition_type
        unique_types = self.test_result_df['transition_type'].unique()
        # 创建自定义图例项
        legend_elements = [
            Line2D([0], [0], color=sns.color_palette("tab10")[i], lw=2, linestyle='-', label=f"{ut} (True)")
            for i, ut in enumerate(unique_types)]
        legend_elements += [
            Line2D([0], [0], color=sns.color_palette("tab10")[i], lw=2, linestyle='--', label=f"{ut} (Pred)")
            for i, ut in enumerate(unique_types)]

        # 添加自定义图例到图形
        self.ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Transition Type")

        self.ax.set_yticks(np.linspace(0, 1, 5))
        self.ax.set_xlim(left=0)
        self.ax.set_ylim([0, 1])
        self.ax.set_xlabel("k")  # 设置x轴标注
        self.ax.set_ylabel("State transition probability")  # 设置y轴标注
        self.ax.grid(True)
        # 显示图形
        plt.tight_layout()

    def draw(self):
        '''
        绘制图像
        :return:
        '''
        self.fig, self.ax = plt.subplots(figsize=(10 / self.shrink_times, 6 / self.shrink_times))
        # 创建一个新的列来表示节点状态迁移的类型
        self.test_result_df['transition_type'] = self.test_result_df.apply(
            lambda row: f"{row['x_lable']} → {row['y_ob_lable']}", axis=1
        )
        # 绘制 trans_prob_true 的实线图（真实数据）
        sns.lineplot(
            # x='adj_act_edges',
            x='adj_act_triangles',
            y='trans_prob_true',
            hue='transition_type',
            data=self.test_result_df,
            ax=self.ax,
            palette="tab10",  # 使用默认的10种颜色调色板
            linestyle="-",  # 实线表示真实值
            linewidth=3,  # 设置线条粗细为 2
            alpha=0.8,  # 设置透明度为 0.8
            legend=False  # 暂时不显示图例，后面一起处理
        )

        # 绘制 trans_prob_pred 的虚线图（预测数据）
        sns.lineplot(
            x='adj_act_edges',
            y='trans_prob_pred',
            hue='transition_type',
            data=self.test_result_df,
            ax=self.ax,
            palette="tab10",  # 使用相同的调色板
            linestyle="--",  # 虚线表示预测值
            linewidth=3,  # 设置线条粗细为 2
            alpha=0.8,  # 设置透明度为 0.8
            legend=False  # 暂时不显示图例，后面一起处理
        )


class FigKLoss(MatplotDrawer):
    def __init__(self,
                 test_result_info,
                 test_result_df,
                 dynamics_STATES_MAP,
                 model_performance_dict, *args, **kwargs):
        super().__init__()
        self.task_name = 'FigKLoss'
        self.config = self.config_drawer['fig_k_loss']
        self.test_result_info = test_result_info
        self.STATES_MAP = dynamics_STATES_MAP
        self.test_result_df = test_result_df
        self.shrink_times = 1.5
        MAX_DIMENSION = self.test_result_info['MAX_DIMENSION']
        if MAX_DIMENSION == 1:
            self.x_lable = 'k_0'
        elif MAX_DIMENSION == 2:
            self.x_lable = 'k_2'

    def edit_ax(self):
        '''
        编辑ax
        :return:
        '''
        # self.ax.set_title(r' $R$ = {:0.5f}'.format(self.corrcoef))
        # self.ax.set_xticks(np.linspace(0,1,5))
        # 自定义图例
        from matplotlib.lines import Line2D
        # 获取唯一的 transition_type
        unique_types = self.test_result_df['transition_type'].unique()
        palette = sns.color_palette("tab10", n_colors=len(unique_types))
        # 创建自定义图例项
        legend_elements = [
            Line2D([0], [0], color=palette[i], lw=2, linestyle='-', label=f"{ut}")
            for i, ut in enumerate(unique_types)
        ]

        # 添加自定义图例到图形, 调整 bbox_to_anchor 防止 legend 超出范围
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), title="Transition Type",
                       fontsize=10)

        self.ax.set_ylim(bottom=0)
        # self.ax.set_ylim([0, 0.007])

        self.ax.set_xlabel(self.x_lable)  # 设置x轴标注
        self.ax.set_ylabel("Loss")  # 设置y轴标注
        self.ax.grid(True)
        # 显示图形
        plt.tight_layout()
        # plt.show()

    def draw(self):
        '''
        绘制图像
        :return:
        '''
        self.fig, self.ax = plt.subplots(figsize=(10 / self.shrink_times, 6 / self.shrink_times))
        # 创建一个新的列来表示节点状态迁移的类型
        self.test_result_df['transition_type'] = self.test_result_df.apply(
            lambda row: f"{row['x_lable']} → {row['y_ob_lable']}", axis=1
        )

        # 绘制 trans_prob_true 的实线图（真实数据）
        sns.lineplot(
            x=self.x_lable,
            y='node_loss',
            hue='transition_type',
            data=self.test_result_df,
            ax=self.ax,
            palette="tab10",  # 使用默认的10种颜色调色板
            linestyle="-",  # 实线表示真实值
            linewidth=3,  # 设置线条粗细为 2
            alpha=0.8,  # 设置透明度为 0.8
            errorbar=None,  # 不绘制区域
            legend=False  # 暂时不显示图例，后面一起处理
        )


class FigKDistribution(MatplotDrawer):
    def __init__(self,
                 test_result_info,
                 test_result_df,
                 dynamics_STATES_MAP,
                 model_performance_dict,
                 network,
                 *args, **kwargs):
        super(FigKDistribution, self).__init__()
        self.task_name = 'FigKDistribution'
        self.config = self.config_drawer['fig_k_distribution']
        self.test_result_info = test_result_info
        self.network = network
        self.shrink_times = 1.5

    def edit_ax(self):
        '''
        编辑ax
        :return:
        '''

        # 显示图形
        plt.tight_layout()
        # plt.show()

    def draw(self):
        '''
        绘制图像
        :return:
        '''

        # 获取度分布
        self.k_0_list = self.network.inc_matrix_adj0.sum(dim=1).to_dense().to('cpu').numpy()

        # 判断 MAX_DIMENSION 的值
        if self.test_result_info['MAX_DIMENSION'] == 1:
            # 只绘制一阶度 k_0_list 的度分布
            self.fig, self.ax = plt.subplots(figsize=(10 / self.shrink_times, 6 / self.shrink_times))
            sns.histplot(self.k_0_list, kde=True, bins=30, color='blue', ax=self.ax)
            self.ax.set_title('Distribution of First-Order Degree (k_0)', fontsize=16)
            self.ax.set_xlabel('First-Order Degree (k_0)', fontsize=14)
            self.ax.set_ylabel('Frequency', fontsize=14)
            self.ax.grid(True)

        elif self.test_result_info['MAX_DIMENSION'] == 2:
            self.k_2_list = self.network.inc_matrix_adj2.sum(dim=1).to_dense().to('cpu').numpy()
            # 创建上下排列的两个子图
            self.fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10 / self.shrink_times, 12 / self.shrink_times))

            # 绘制一阶度 k_0_list 的度分布
            sns.histplot(self.k_0_list, kde=True, bins=30, color='blue', ax=ax1)
            ax1.set_title('Distribution of First-Order Degree (k_0)', fontsize=16)
            ax1.set_xlabel('First-Order Degree (k_0)', fontsize=14)
            ax1.set_ylabel('Frequency', fontsize=14)
            ax1.set_ylabel('Frequency', fontsize=14)
            ax1.grid(True)

            # 绘制二阶度 k_2_list 的度分布
            sns.histplot(self.k_2_list, kde=True, bins=30, color='green', ax=ax2)
            ax2.set_title('Distribution of Second-Order Degree (k_2)', fontsize=16)
            ax2.set_xlabel('Second-Order Degree (k_2)', fontsize=14)
            ax2.set_ylabel('Frequency', fontsize=14)
            ax2.set_ylabel('Frequency', fontsize=14)
            ax2.grid(True)


class FigTimeEvolution(MatplotDrawer):
    def __init__(self,
                 test_result_info,
                 test_result_df,
                 dynamics_STATES_MAP,  # 动态传入 STATES_MAP
                 *args, **kwargs):
        super(FigTimeEvolution, self).__init__()
        self.config = self.config_drawer['fig_time_evolution']
        self.task_name = 'FigTimeEvolution'
        self.test_result_info = test_result_info
        self.test_result_df = test_result_df
        self.dynamics_STATES_MAP = dynamics_STATES_MAP
        self.shrink_times = 1.5

    def edit_ax(self):
        '''
        编辑ax
        :return:
        '''
        # 设置图像标签和标题
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Density')
        self.ax.set_title('Time Evolution of Dynamics (Original vs ML)')

        # 显示图例
        self.ax.legend(fontsize='small', markerscale=0.5,loc="upper right")

        # 调整图像布局
        plt.tight_layout()

    def draw(self):
        '''
        绘制时间演化图像
        '''
        # 提取列名，适配不同的动力学模型状态
        state_columns_map = {}
        for state_name, state_id in self.dynamics_STATES_MAP.items():
            ori_cols = [col for col in self.test_result_df.columns if
                        col.startswith(f'ori_x0_T') and col.endswith(f'_{state_name}')]
            ml_cols = [col for col in self.test_result_df.columns if
                       col.startswith(f'ml_x0_T') and col.endswith(f'_{state_name}')]
            state_columns_map[state_name] = {'ori': ori_cols, 'ml': ml_cols}

        # 设置图像大小
        self.fig, self.ax = plt.subplots(figsize=(10 / self.shrink_times, 6 / self.shrink_times))
        colors = sns.color_palette("deep", len(self.dynamics_STATES_MAP))  # 使用不同颜色，动态适配状态数量

        # 绘制不同状态的原始和机器学习动力学数据
        for i, (state_name, cols) in enumerate(state_columns_map.items()):
            for col in cols['ori']:  # 绘制原始数据
                sns.lineplot(data=self.test_result_df, x=self.test_result_df.index, y=col, label=f'{col}_ori',
                             ax=self.ax,
                             linewidth=2,
                             color=colors[i])
            for col in cols['ml']:  # 绘制机器学习数据
                sampled_df = self.test_result_df.iloc[::3, :]  # 抽样
                sns.scatterplot(data=sampled_df, x=sampled_df.index, y=col, label=f'{col}_ml',
                                ax=self.ax,
                                color=colors[i],
                                s=85,  # 散点大小
                                alpha=0.5)
