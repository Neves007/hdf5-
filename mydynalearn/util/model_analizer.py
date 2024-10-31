import torch
from tqdm.autonotebook import tqdm
import numpy as np
import torch.nn.functional as F

from mydynalearn.dataset import graphDataSetLoader
class ModelAnalizer():
    def __init__(self,exp,test_set):
        self.gnn_Model = exp.model
        self.gnn_Model.load_state_dict(
            torch.load(exp.gnnExpeiment_Config.datapath_to_model + "BestEpoch.pkl"))
        self.DEVICE = exp.gnnExpeiment_Config.torch_DEVICE
        self.test_set = test_set
        # S_true_TP_degree, S_predict_TP_degree, I_true_TP_degree, I_predict_TP_degree = self.analyze_dgree_TP()
        # S_true_TP, S_predict_TP, I_true_TP_alltime, I_predict_TP_alltime = self.analyze_pre_target_y()

    def __get_degree(self, test_loader):
        adjList_Dict = test_loader[0][4]["adjList_Dict"]
        Maxdegree = adjList_Dict["1-simplex"]["adjListValid"].shape[1]
        return Maxdegree
    def analyze_dgree_TP(self):
        test_loader = graphDataSetLoader(self.test_set)

        Maxdegree = self.__get_degree(test_loader)
        S_true_TP_degree = [[] for _ in range(Maxdegree+1)]
        S_predict_TP_degree = [[] for _ in range(Maxdegree+1)]
        I_true_TP_degree = [[] for _ in range(Maxdegree+1)]
        I_predict_TP_degree = [[] for _ in range(Maxdegree+1)]

        process_bar = tqdm(
            test_loader,
            maxinterval=10,
            mininterval=2,
            bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}',
            total=test_loader.data_set['x'].shape[0],
        )
        for time_index, test_dataset_per_time in enumerate(process_bar):
            self.gnn_Model.eval()
            x, y_ob, true_TP, num_activate_simplex_Dict, structure_info = test_dataset_per_time
            logsoftmax_output = self.gnn_Model(x, structure_info)

            with torch.no_grad():
                predict_TP = torch.exp(logsoftmax_output)

                S_nodes_index = torch.where(x==0)[0]
                I_nodes_index = torch.where(x==1)[0]


                S_true_TP = true_TP[S_nodes_index]
                S_predict_TP = predict_TP[S_nodes_index]
                S_num_ac_1simplex = num_activate_simplex_Dict['1-simplex'][S_nodes_index]
                I_true_TP = true_TP[I_nodes_index]
                I_predict_TP = predict_TP[I_nodes_index]
                I_num_ac_1simplex = num_activate_simplex_Dict['1-simplex'][I_nodes_index]
                for i,S_node in enumerate(S_nodes_index):
                    S_num_ac_1simplex_pernode = S_num_ac_1simplex[i]
                    S_true_TP_pernode = S_true_TP[i].cpu().numpy()
                    S_predict_TP_pernode = S_predict_TP[i].cpu().numpy()
                    S_true_TP_degree[int(S_num_ac_1simplex_pernode.data.item())].append(S_true_TP_pernode)
                    S_predict_TP_degree[int(S_num_ac_1simplex_pernode.data.item())].append(S_predict_TP_pernode)
                for i, I_node in enumerate(I_nodes_index):
                    I_num_ac_1simplex_pernode = I_num_ac_1simplex[i]
                    I_true_TP_pernode = I_true_TP[i].cpu().numpy()
                    I_predict_TP_pernode = I_predict_TP[i].cpu().numpy()
                    I_true_TP_degree[int(I_num_ac_1simplex_pernode.data.item())].append(I_true_TP_pernode)
                    I_predict_TP_degree[int(I_num_ac_1simplex_pernode.data.item())].append(I_predict_TP_pernode)
        process_bar.close()
        return S_true_TP_degree, S_predict_TP_degree, I_true_TP_degree, I_predict_TP_degree

    def analyze_pre_target_y(self):
        test_loader = graphDataSetLoader(self.test_set)
        process_bar = tqdm(
            test_loader,
            maxinterval=1,
            mininterval=1,
            bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}',
            total=100,
        )
        for time_index, test_dataset_per_time in enumerate(process_bar):
            self.gnn_Model.eval()
            x, weight, y_ob, true_TP, num_activate_simplex_Dict, structure_info = test_dataset_per_time
            logsoftmax_output = self.gnn_Model(x, structure_info)
            predict_TP = torch.exp(logsoftmax_output)
            with torch.no_grad():
                S_nodes_index = torch.where(x == 0)[0]
                I_nodes_index = torch.where(x == 1)[0]

                S_true_TP = true_TP[S_nodes_index].cpu().numpy()
                S_predict_TP = predict_TP[S_nodes_index].cpu().numpy()
                I_true_TP = true_TP[I_nodes_index].cpu().numpy()
                I_predict_TP = predict_TP[I_nodes_index].cpu().numpy()
        return S_true_TP, S_predict_TP, I_true_TP, I_predict_TP


