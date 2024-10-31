import torch
from scipy.stats import binom
import numpy as np
'''
公式（7）和公式（16），计算Loss前面的那个权重的。
避免训练数据不平衡导致训练不稳定。
'''


import torch

class SimpleDynamicWeight():
    def __init__(self,DEVICE,old_x0,new_x0,**kwargs):
        assert len(kwargs) == 0
        self.DEVICE = DEVICE
        self.old_x0 = old_x0
        self.new_x0 = new_x0

    def get_weight(self):

        # 将 one-hot 编码转换为单个整数标签，方便统计
        old_labels = torch.argmax(self.old_x0, dim=1)
        new_labels = torch.argmax(self.new_x0, dim=1)

        num_states = self.old_x0.shape[1]
        # 计算迁移类别（例如，从状态3到状态1可以编码为 3*4+1 = 13）
        transitions = old_labels * num_states + new_labels

        # 计算每种迁移的发生频率
        transition_counts = torch.zeros(num_states*num_states,device=self.DEVICE)  # 因为有16种可能的迁移（从0到15）
        for t in transitions:
            transition_counts[t] += 1
        transition_probs = transition_counts / transition_counts.sum()

        # 将每个节点的迁移映射到其对应的频率
        node_transition_probs = transition_probs[transitions]

        # 计算映射参数
        min_val = node_transition_probs.min()
        max_val = node_transition_probs.max()
        a = (1 - 2) / (max_val - min_val)
        b = 2 - a * min_val

        # 应用映射
        mapped_transition_probs = a * node_transition_probs + b

        return mapped_transition_probs
