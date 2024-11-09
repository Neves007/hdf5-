import torch
import numpy as np
import re
def feature_to_lable(feature,STATES):
    lable = [STATES[index] for index in feature.argmax(1)]
    return lable

def get_transtype_latex(origion_state:str, target_state:str):
    '''
    将迁移转为latex
    :param origion_state:
    :param target_state:
    :return:
    '''

    latex = "$\\mathrm{{{} \\: \\rightarrow \\: {}}}$".format(origion_state,target_state )
    return latex

def extract_trans_prob(y_ob_T, prob_map):
    """
    根据观测状态的编码y_ob_T，提取出对应状态的迁移概率
    :param y_ob_T: 状态编码，可以是 numpy 数组或 torch 张量
    :param prob_map: 迁移概率映射，嵌套字典或嵌套列表
    :return: 对应的迁移概率数组
    """
    if isinstance(y_ob_T, np.ndarray):
        # 使用 numpy 操作
        state_indices = np.argmax(y_ob_T, axis=1)
        return np.array([prob_map[node_index][state_index] for node_index, state_index in enumerate(state_indices)])
    elif isinstance(y_ob_T, torch.Tensor):
        # 使用 torch 操作
        state_indices = torch.argmax(y_ob_T, dim=1).tolist()
        return np.array([prob_map[node_index][state_index] for node_index, state_index in enumerate(state_indices)])
    else:
        raise TypeError("ori_y_ob_T must be either a numpy array or a torch tensor")