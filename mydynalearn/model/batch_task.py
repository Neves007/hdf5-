from mydynalearn.model.nn.nnlayers import *
import os
import pickle
import torch.nn as nn
import torch
from mydynalearn.model.optimizer import get as get_optimizer
from mydynalearn.model.util import *
import copy
from mydynalearn.logger.logger import *
from tqdm import tqdm


class BatchTask():
    def __init__(self, config):
        self.config = config
        self.model_config = config.model
        self.DEVICE = config.DEVICE

        # model params_dict
        self.IS_WEIGHT = config.IS_WEIGHT

    def weighted_cross_entropy(self,y_true, y_pred, weights=None):
        y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
        loss = weights * (-y_true * torch.log(y_pred)).sum(-1)
        return loss.sum()

    def _do_batch_(self, attention_model, network, dynamics, dataset_per_time):
        '''batch
        :param dataset_per_time:
            - x0, y_ob, y_true, weight
            - 来自DynamicDataset.__getitem__
        :return:
        '''
        x0, y_pred, y_true, y_ob, w = self.prepare_output(attention_model, network, dynamics, dataset_per_time)
        loss = self.weighted_cross_entropy(y_ob, y_pred, w)
        return loss, x0, y_pred, y_true, y_ob, w

    def prepare_output(self, attention_model, network, dynamics, dataset_per_time):
        x0, y_pred, y_true, y_ob, weight = attention_model(network, dynamics, *dataset_per_time)
        if self.IS_WEIGHT == False:
            weight = torch.ones(x0.shape[0]).to(self.DEVICE)
        return x0, y_pred, y_true, y_ob, weight

