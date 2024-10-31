from mydynalearn.model.nn.nnlayers import *
from time import sleep
import os
import pickle
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mydynalearn.model.optimizer import get as get_optimizer
from mydynalearn.drawer import VisdomController
from mydynalearn.model.util import *
import copy
from mydynalearn.logger.logger import *
from tqdm import tqdm
from mydynalearn.model.getter import get as get_attmodel
from mydynalearn.model.batch_task import BatchTask
from mydynalearn.logger import Log

class EpochTasks():
    def __init__(self, config):
        self.config = config
        self.logger = Log("EpochTasks")
        self.EPOCHS = config.model.EPOCHS
        self.need_to_train = self.is_need_to_train()
        self.batch_task = BatchTask(config)
        #
        self.attention_model = get_attmodel(self.config)
        self.get_optimizer = get_optimizer(config.optimizer)
        self.optimizer = self.get_optimizer(self.attention_model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=4, eps = 1e-8, threshold =0.1)

    def get_fileName_model_state_dict(self,epoch_index):
        fileName_model_state_dict = self.config.modelparams_dir_path + "/epoch{:d}_model_state_dict.pth".format(
            epoch_index)
        return fileName_model_state_dict

    def save(self, attention_model, optimizer, epoch_index):
        model_state_dict_file_path = self.get_fileName_model_state_dict(epoch_index)
        self.logger.increase_indent()
        self.logger.log(f"save: {model_state_dict_file_path}")
        torch.save({
            # 存储 batch的state_dict
            'model_state_dict': attention_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_state_dict_file_path)
        self.logger.decrease_indent()

    def load(self, epoch_index):
        '''
        输入：当前epoch_index
        规则：加载模型参数
        输出：
        '''
        self.epoch_index = epoch_index
        checkpoint = torch.load(self.get_fileName_model_state_dict(epoch_index), weights_only=True)
        self.attention_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def is_need_to_trian(self,epoch_index):
        '''
        输入：当前epoch_index
        规则：如果存在model_state_dict文件就不需要训练，否则需要
        输出：是否需要训练
        '''
        fileName_model_state_dict = self.get_fileName_model_state_dict(epoch_index)
        is_need_to_trian = not os.path.exists(fileName_model_state_dict)
        return is_need_to_trian

    def low_the_lr(self ,epoch_index):
        if (epoch_index>0) and (epoch_index % 5 == 0):
            self.optimizer.param_groups[0]['lr'] *= 0.5


    def pack_batch_data(self, epoch_index, time_index, loss, x, y_pred, y_true, y_ob, w):
        data = {'epoch_index': epoch_index,
                'time_index': time_index,
                'loss': loss.cpu(),
                'acc': get_acc(y_ob, y_pred).cpu(),
                'x': x.cpu(),
                'y_pred': y_pred.cpu(),
                'y_true': y_true.cpu(),
                'y_ob': y_ob.cpu(),
                'w': w.cpu(),
                }
        return data

    def is_need_to_train(self):
        '''
        判断该epoch_tasks是否该重新训练
        - 若存在epoch未被训练则需重新训练

        :return: bool
        '''
        tag = False
        for epoch_id in range(self.EPOCHS):
            epoch_model_file = self.get_fileName_model_state_dict(epoch_id)
            if not os.path.exists(epoch_model_file):
                tag =  True
        return tag


    def train_epoch(self,train_set, val_set, network, dynamics, epoch_index):
        self.logger.increase_indent()
        self.logger.log(f"train epoch: {epoch_index}")
        R_stack = []
        for time_index, (train_dataset_per_time, val_dataset_per_time) in enumerate(zip(train_set, val_set)):
            self.attention_model.train()
            self.optimizer.zero_grad()
            train_loss, train_x, train_y_pred, train_y_true, train_y_ob, train_w = self.batch_task._do_batch_(self.attention_model,
                                                                                                              network,
                                                                                                              dynamics,
                                                                                                              train_dataset_per_time)
            train_loss.backward()
            self.optimizer.step()
            self.attention_model.eval()
            val_loss, val_x, val_y_pred, val_y_true, val_y_ob, val_w = self.batch_task._do_batch_(self.attention_model,
                                                                                                  network,
                                                                                                  dynamics,
                                                                                                  val_dataset_per_time)
            val_data = self.pack_batch_data(epoch_index,
                                            time_index,
                                            val_loss,
                                            val_x,
                                            val_y_pred,
                                            val_y_true,
                                            val_y_ob,
                                            val_w)
            y_true_flat = val_y_true.flatten()
            y_pred_flat = val_y_pred.flatten()
            R = torch.corrcoef(torch.stack([y_true_flat, y_pred_flat]))[0, 1]
            item_info = 'Epoch:{:d} LR:{:f} R:{:f}'.format(epoch_index,
                                                           self.optimizer.param_groups[0]['lr'],
                                                           R)
            # self.visdom_drawer.visdomDrawBatch(val_data)
        self.logger.decrease_indent()
    def run_all(self,network, dynamics, train_set, val_set,*args,**kwargs):
        # self.visdom_drawer = VisdomController(self.config, dynamics)
        for epoch_index in range(self.EPOCHS):
            self.train_epoch(train_set,
                             val_set,
                             network,
                             dynamics,
                             epoch_index)
            self.save(self.attention_model, self.optimizer, epoch_index)
            self.low_the_lr(epoch_index)

    def run_test_epoch(self, network, dynamics, test_set,*args,**kwargs):
        self.logger.increase_indent()
        self.logger.log("run test epoch")
        self.attention_model.eval()
        for time_index,test_dataset_per_time in enumerate(test_set):
            test_loss, test_x, test_y_pred, test_y_true, test_y_ob, test_w = self.batch_task._do_batch_(self.attention_model,
                                                                                                        network,
                                                                                                        dynamics,
                                                                                                        test_dataset_per_time)
            test_data = self.pack_batch_data(self.epoch_index, time_index, test_loss, test_x, test_y_pred, test_y_true, test_y_ob,
                                             test_w)
            yield test_data
        self.logger.decrease_indent()

