from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
from mydynalearn.analyze.utils.performance_data.utils import _get_metrics

class FigMaxR():
    def __init__(self,config,dynamics):
        self.config = config
        self.colors = ["b", "orange"]
        self.markers = ['.', ',', 'o', 'v', '^', '<', '>', 's', 'p', '*', '+', 'x']
        self.exp_NAME_list = ["dynamicLearning-ER-CompUAU-DiffSAT",
            "dynamicLearning-ER-CompUAU-GAT",
            "dynamicLearning-ER-CompUAU-SAT",
            "dynamicLearning-ER-UAU-DiffSAT",
            "dynamicLearning-ER-UAU-GAT",
            "dynamicLearning-ER-UAU-SAT",
            "dynamicLearning-SCER-SCCompUAU-DiffSAT",
            "dynamicLearning-SCER-SCCompUAU-GAT",
            "dynamicLearning-SCER-SCCompUAU-SAT",
            "dynamicLearning-SCER-SCUAU-DiffSAT",
            "dynamicLearning-SCER-SCUAU-GAT",
            "dynamicLearning-SCER-SCUAU-SAT"]
        assert len(self.exp_NAME_list)==len(self.markers)
        self.label = None

    def edit_ax(self,epoch_index,performance_data):
        corrcoef,r2 = _get_metrics(performance_data)
        self.ax.set_title(r'epoch = {:d}, $R$ = {:0.5f}'.format(epoch_index, corrcoef))
        self.ax.set_xticks(np.linspace(0,1,5))
        self.ax.set_yticks(np.linspace(0,1,5))
        self.ax.set_xlim([0,1])
        self.ax.set_ylim([0,1])
        self.ax.set_xlabel("Target")  # 设置x轴标注
        self.ax.set_ylabel("prediction")  # 设置y轴标注

        self.legend_elements = self.get_legend_elements()
        self.ax.legend(handles=self.legend_elements, labels=self.label)
        self.ax.grid(True)

    def scatter(self, ax,max_R):
        max_R_index,stable_r_value = max_R
        exp_NAME = self.config.NAME
        exp_index = self.exp_NAME_list.index(exp_NAME)
        ax.scatter(x=max_R_index,
                   y=stable_r_value,
                   c=self.colors[self.config.IS_WEIGHT],
                   marker=self.markers[exp_index],
                   s=50,
                   alpha=0.8)

    def get_legend_elements(self):
        legend_elements = [plt.scatter([0],
                                       [0],
                                       c=self.colors[index],
                                       marker=self.markers[index],
                                       s=53,
                                       alpha=0.8) for index in range(len(self.label))]
        return legend_elements