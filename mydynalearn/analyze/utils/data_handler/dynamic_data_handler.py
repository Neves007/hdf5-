import torch
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score,log_loss
from mydynalearn.transformer.transformer import transition_to_latex
import re
from ..utils import get_all_transition_types
from mydynalearn.transformer.transformer import transition_to_latex
from sklearn.metrics import confusion_matrix

class DynamicDataHandler():
    def __init__(self, dynamics, performance_result_generator, **kwargs):
        self.STATES_MAP = dynamics.STATES_MAP
        self.dynamics = dynamics
        self.performance_result_generator = performance_result_generator
        self.source_data = {
            "x": [],
            "y_ob": [],
            "y_true": [],
            "y_pred": [],
        }


    def _get_dict_k(self):
        '''
        以字典形式合并所有节点的度（一阶度和二阶度）
        :return: 度字典
        '''
        network = self.dynamics.network
        MAX_DIMENSION = network.MAX_DIMENSION
        inc_matrix_adj_info = network.inc_matrix_adj_info
        k_data = {}
        for dimention_index in range(MAX_DIMENSION+1):
            k_data_key = "k_"+str(dimention_index)
            # 度
            inc_matrix_adj_info_key = list(inc_matrix_adj_info.keys())[dimention_index]
            k = torch.sparse.sum(inc_matrix_adj_info[inc_matrix_adj_info_key],dim=1)
            k = k.to_dense().cpu().detach().numpy()
            k_data[k_data_key] = k
            # 邻居状态
        return k_data

    def _get_node_loss(self, y_pred, y_true):
        # 均方差（不能使用交叉熵）
        node_loss = np.mean((y_pred - y_true) ** 2,axis=1)  # 按照需要的维度求和
        return node_loss

    def _get_dict_neighbor_status_dict(self, x):
        '''
        以字典形式合并所有节点的邻居状态数量（一阶和二阶）
        :return: 邻居状态数量
        '''
        #todo: 在dynamic中加入_get_dict_neighbor_status
        tensor_x = torch.tensor(x,device=self.dynamics.DEVICE)
        self.dynamics.set_features(tensor_x)
        adj_activate_simplex_dict = self.dynamics.get_adj_activate_simplex_dict()
        return adj_activate_simplex_dict
    
    def _get_source_data(self, test_result):
        '''
        获取源数据
        :return:
        '''

        x = test_result["x"].cpu().detach().numpy()
        y_ob = test_result["y_ob"].cpu().detach().numpy()
        y_true = test_result["y_true"].cpu().detach().numpy()
        y_pred = test_result["y_pred"].cpu().detach().numpy()
        w_T = test_result["w"].cpu().detach().numpy()
        sub_source_data = {
            "x": x,
            "y_ob": y_ob,
            "y_true": y_true,
            "y_pred": y_pred,
            "weight": w_T,
        }
        return sub_source_data

    def _get_result_data(self,sub_source_data):
        '''
        获取performance data 分析所需的数据
        :param test_result:
        :return: 聚集的字典
        '''

        # 测试结果数据
        x = sub_source_data["x"]
        y_ob = sub_source_data["y_ob"]
        y_true = sub_source_data["y_true"]
        y_pred = sub_source_data["y_pred"]
        w_T = sub_source_data["weight"]
        state_map = list(self.STATES_MAP.keys())

        test_result_data = {
            "node_id" : np.arange(x.shape[0]),
            "weight":w_T,
            "x_lable": self.dynamics.extract_state_labels(x, state_map),
            "y_ob_lable": self.dynamics.extract_state_labels(y_ob, state_map),
            "y_pred_lable": np.array([np.random.choice(state_map, p=y_pred[i]) for i in range(len(y_pred))]),
            "trans_prob_true": self.dynamics.extract_trans_prob(y_ob, y_true),
            "trans_prob_pred": self.dynamics.extract_trans_prob(y_ob, y_pred),
            "node_loss": self._get_node_loss(y_pred, y_true), # 真实值与预测值的loss而不是训练的loss
        }
        # 网络结构数据，（节点度）
        k_data = self._get_dict_k()
        # 邻居激活态数据
        adj_activate_simplex_data = self._get_dict_neighbor_status_dict(x)
        # 迁移类型数据
        trans_type_data = {
            "true_trans_type": self.dynamics.generate_trans_type(test_result_data["x_lable"], test_result_data["y_ob_lable"]),
            "pred_trans_type": self.dynamics.generate_trans_type(test_result_data["x_lable"], test_result_data["y_pred_lable"]),
        }


        # 将几个数据合并
        merged_dict = {
            **test_result_data,
            **k_data,
            **adj_activate_simplex_data,
            **trans_type_data,
            }


        return merged_dict,sub_source_data



    def _get_evolution_result_data(self,sub_source_data):
        '''
        将test_result转为numpy，以字典的形式聚集起来
        :param test_result:
        :return: 聚集的字典
        '''

        # 测试结果数据
        x = sub_source_data["x"]
        y_ob = sub_source_data["y_ob"]
        y_true = sub_source_data["y_true"]
        y_pred = sub_source_data["y_pred"]
        w_T = sub_source_data["w"]
        state_map = list(self.STATES_MAP.keys())

        test_result_data = {
            "node_id" : np.arange(x.shape[0]),
            "weight":w_T,
            "x_lable": self.dynamics.extract_state_labels(x, state_map),
            "y_ob_lable": self.dynamics.extract_state_labels(y_ob, state_map),
            "y_pred_lable": np.array([np.random.choice(state_map, p=y_pred[i]) for i in range(len(y_pred))]),
            "trans_prob_true": self.dynamics.extract_trans_prob(y_ob, y_true),
            "trans_prob_pred": self.dynamics.extract_trans_prob(y_ob, y_pred),
            "node_loss": self._get_node_loss(y_pred, y_true), # 真实值与预测值的loss而不是训练的loss
        }
        # 网络结构数据，（节点度）
        k_data = self._get_dict_k()
        # 邻居激活态数据
        adj_activate_simplex_data = self._get_dict_neighbor_status_dict(x)
        # 迁移类型数据
        trans_type_data = {
            "true_trans_type": self.dynamics.generate_trans_type(test_result_data["x_lable"], test_result_data["y_ob_lable"]),
            "pred_trans_type": self.dynamics.generate_trans_type(test_result_data["x_lable"], test_result_data["y_pred_lable"]),
        }


        # 将几个数据合并
        merged_dict = {
            **test_result_data,
            **k_data,
            **adj_activate_simplex_data,
            **trans_type_data,
            }


        return merged_dict,sub_source_data


    def append_source_data(self,sub_source_data):
        for key in self.source_data.keys():
            self.source_data[key].extend(sub_source_data[key])

    def get_result_dataframe(self, test_result_info):
        '''
        将把所有测试结构转化为dataframe，合并起来
        :return: dataframe
        '''
        df = pd.DataFrame()
        for test_id,test_result in enumerate(self.performance_result_generator):
            # 一个test数据
            sub_source_data = self._get_source_data(test_result)
            dict_data, sub_source_data = self._get_result_data(sub_source_data)
            self.append_source_data(sub_source_data)
            temp_df = pd.DataFrame(dict_data)
            # dict to dataframe
            df = pd.concat([df, temp_df],ignore_index=True)

        df = df.assign(**test_result_info)
        return df

    def get_evolution_result_dataframe(self, test_result_info):
        '''
        将把所有测试结构转化为dataframe，合并起来
        :return: dataframe
        '''
        df = pd.DataFrame()
        for test_id,test_result in enumerate(self.performance_result_generator):
            # 一个test数据
            sub_source_data = self._get_source_data(test_result)
            dict_data, sub_source_data = self._get_result_data(sub_source_data)
            self.append_source_data(sub_source_data)
            temp_df = pd.DataFrame(dict_data)
            # dict to dataframe
            df = pd.concat([df, temp_df],ignore_index=True)

        df = df.assign(**test_result_info)
        return df


    def get_model_performance(self,test_result_df):
        '''
        通过self.source_data计算模型的性能
            混淆矩阵，F1 分数，多类别 AUC
        :return:
        '''
        x = np.array(self.source_data["x"])
        y_ob = np.array(self.source_data["y_ob"])
        y_true = np.array(self.source_data["y_true"])
        y_pred = np.array(self.source_data["y_pred"])

        # Convert predicted probabilities to predicted labels
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        # 计算交叉熵损失
        epsilon = 1e-15  # a small value to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # ensure y_pred is within (epsilon, 1 - epsilon)
        cross_loss = (-y_true * np.log(y_pred)).sum(axis=1).mean()

        # 计算混淆矩阵
        confusion_matrix_df = pd.crosstab(test_result_df['true_trans_type'],
                                          test_result_df['pred_trans_type'],
                                          rownames=['pred_trans_type'],
                                          colnames=['true_trans_type'],
                                          normalize=False,
                                          )
        all_transition_types = get_all_transition_types(self.STATES_MAP)
        all_labels = [transition_to_latex(origion_state, target_state) for origion_state, target_state in
                      all_transition_types]
        confusion_matrix_df = confusion_matrix_df.reindex(index=all_labels, columns=all_labels, fill_value=0.)
        confusion_matrix_df = confusion_matrix_df.div(confusion_matrix_df.sum(axis=1), axis=0)
        confusion_matrix_df = confusion_matrix_df.fillna(0)
        # 计算F1分数
        f1 = f1_score(y_true_labels, y_pred_labels, average='macro')
        y_true_value = (y_ob * y_true).sum(axis=1)
        y_pred_value = (y_ob * y_pred).sum(axis=1)
        # R值
        R = np.corrcoef([y_true_value,y_pred_value])[0,1]
        model_performance_dict = {
            "cm": confusion_matrix_df,
            "f1": f1,
            "R": R,
            "cross_loss":cross_loss
        }
        return model_performance_dict


