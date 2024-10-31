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

class DynamicDataHandlerTimeEvolution():
    def __init__(self,network, dynamics, dynamic_dataset_time_evolution,**kwargs):
        self.network = network
        self.STATES_MAP = dynamics.STATES_MAP
        self.dynamics = dynamics
        self.dynamic_dataset_time_evolution = dynamic_dataset_time_evolution


    def get_evolution_result_dataframe(self, test_result_info):
        '''
        将把所有测试结构转化为dataframe，合并起来
        :return: dataframe
        '''
        df = pd.DataFrame()
        for key_i in self.dynamic_dataset_time_evolution:
            for key_j in self.STATES_MAP:
                state_index = self.STATES_MAP[key_j]
                new_value = self.dynamic_dataset_time_evolution[key_i][:,state_index]
                data = {"_".join([key_i,key_j]): new_value.detach().cpu().numpy()}
                df = df.assign(**data)
        return df

