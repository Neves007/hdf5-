from visdom import Visdom
import torch
import numpy as np
from mydynalearn.drawer.utils import *
from mydynalearn.drawer.visdom_drawer.visdom_batch_drawer.visdom_batch_drawer import VisdomBatchDrawer
class VisdomBatchDrawerCompUAU(VisdomBatchDrawer):
    def __init__(self,dynamics) -> None:
        super().__init__()
        self.STATES_MAP = dynamics.STATES_MAP
        self.wind = Visdom()

        self.scatter_colors = (self.COLORS["red"],
                               self.COLORS["orange"],
                               self.COLORS["yellow"],
                               self.COLORS["green"],
                               self.COLORS["cyan"],
                               self.COLORS["blue"],
                               self.COLORS["purple"])

        # 初始化窗口参数



    def get_performance_data(self, data):
        epoch_index, time_index, loss, acc, x, y_pred, y_true, y_ob, w = unpackBatchData(
            data)
        with torch.no_grad():
            U_U = torch.where((x[:, self.STATES_MAP["UU"]] == 1) & (y_ob[:,self.STATES_MAP["UU"]] == 1))[0]
            U_A1 = torch.where((x[:, self.STATES_MAP["UU"]] == 1) & (y_ob[:,self.STATES_MAP["A1U"]] == 1))[0]
            U_A2 = torch.where((x[:, self.STATES_MAP["UU"]] == 1) & (y_ob[:,self.STATES_MAP["UA2"]] == 1))[0]
            A1_A1 = torch.where((x[:, self.STATES_MAP["A1U"]] == 1) & (y_ob[:,self.STATES_MAP["A1U"]] == 1))[0]
            A1_U = torch.where((x[:, self.STATES_MAP["A1U"]] == 1) & (y_ob[:,self.STATES_MAP["UU"]] == 1))[0]
            A2_A2 = torch.where((x[:, self.STATES_MAP["UA2"]] == 1) & (y_ob[:, self.STATES_MAP["UA2"]] == 1))[0]
            A2_U = torch.where((x[:, self.STATES_MAP["UA2"]] == 1) & (y_ob[:, self.STATES_MAP["UU"]] == 1))[0]

            U_U_true_pred = torch.cat((y_true[U_U,self.STATES_MAP["UU"]].view(-1, 1), y_pred[U_U,self.STATES_MAP["UU"]].view(-1, 1)), dim=1).cpu().numpy()
            U_A1_true_pred = torch.cat((y_true[U_A1,self.STATES_MAP["A1U"]].view(-1, 1), y_pred[U_A1,self.STATES_MAP["A1U"]].view(-1, 1)), dim=1).cpu().numpy()
            U_A2_true_pred = torch.cat((y_true[U_A2,self.STATES_MAP["UA2"]].view(-1, 1), y_pred[U_A2,self.STATES_MAP["UA2"]].view(-1, 1)), dim=1).cpu().numpy()
            A1_A1_true_pred = torch.cat((y_true[A1_A1,self.STATES_MAP["A1U"]].view(-1, 1), y_pred[A1_A1,self.STATES_MAP["A1U"]].view(-1, 1)), dim=1).cpu().numpy()
            A1_U_true_pred = torch.cat((y_true[A1_U,self.STATES_MAP["UU"]].view(-1, 1), y_pred[A1_U,self.STATES_MAP["UU"]].view(-1, 1)), dim=1).cpu().numpy()
            A2_A2_true_pred = torch.cat((y_true[A2_A2,self.STATES_MAP["UA2"]].view(-1, 1), y_pred[A2_A2,self.STATES_MAP["UA2"]].view(-1, 1)), dim=1).cpu().numpy()
            A2_U_true_pred = torch.cat((y_true[A2_U,self.STATES_MAP["UU"]].view(-1, 1), y_pred[A2_U,self.STATES_MAP["UU"]].view(-1, 1)), dim=1).cpu().numpy()

        performance_data = [U_U_true_pred,
                            U_A1_true_pred,
                            U_A2_true_pred,
                            A1_A1_true_pred,
                            A1_U_true_pred,
                            A2_A2_true_pred,
                            A2_U_true_pred]
        self.performance_data_null_filtering(performance_data)
        performance_data_type = [((index+1)*np.ones(data.shape[0])).astype(np.int16)
                                 for index,data in enumerate(performance_data)]
        legend = ["U to U",
            "U to A1",
            "U to A2",
            "A1 to A1",
            "A1 to U",
            "A2 to A2",
            "A2 to U"]
        return performance_data,performance_data_type,legend

    def draw_performance(self,val_data,update = 'append'):
        '''

        :param val_data:
        :param update: 'append' to append data, 'replace' to use new data,
        :return:
        '''
        data = val_data
        performance_data,performance_data_type,legend = self.get_performance_data(data)
        X = np.concatenate(performance_data)
        Y = np.concatenate(performance_data_type)
        colors = np.concatenate(self.scatter_colors, axis=0)

        corrcoef = np.corrcoef(X.T)[0,1]

        opts = dict(title='Train performance, r = {:0.4f}'.format(corrcoef),
                    legend=legend,
                    xtickmin=0, # 坐标设置
                    xtickmax=1,
                    xtickstep=0.1,
                    ytickmin=0,
                    ytickmax=1,
                    ytickstep=0.1,
                    markercolor=colors,
                    markersize=5,
                    )
        self.wind.scatter(X=X,Y=Y, win='Train performance', opts=opts)