import os.path
import torch
from mydynalearn.logger import Log
from mydynalearn.util.lazy_loader import *
from mydynalearn.util.replication import Replication

class DynamicDatasetTimeEvolutionML(TorchLazyLoader, Replication):
    '''数据集类
    通过网络network和dynamics来说生成动力学数据集

    生成数据：
        - run_dynamic_process 连续时间动力学数据
        - run 生成动力学数据
    '''
    def __init__(self, exp, ml_model, network, dynamics) -> None:
        self.config = exp.config
        self.logger = Log("DynamicDatasetTimeEvolution")
        self.exp = exp
        self.ml_model = ml_model
        self.DEVICE = self.config.DEVICE
        self.network = network
        self.dynamics = dynamics
        self.TIME_EVOLUTION_STEPS = self.exp.config.dataset.TIME_EVOLUTION_STEPS
        self.dataset_file_path = self._get_dataset_file_path()
        TorchLazyLoader.__init__(self,self.dataset_file_path)
        Replication.__init__(self,replication_data_name_list=["data"],n=50)
        self._init_dataset()


    def _get_dataset_file_path(self):
        dataset_file_name = f"ML_EVOLUTION_DATASET_{self.network.NAME}_{self.dynamics.NAME}_{self.exp.model.NAME}_epoch{self.ml_model.epoch_index}.pth"
        return os.path.join(self.config.ml_time_evolution_dataset_dir_path, dataset_file_name)

    def _init_dataset(self):
        assert self.network.MAX_DIMENSION == self.dynamics.MAX_DIMENSION
        NUM_NODES = self.network.NUM_NODES
        NUM_STATES = self.dynamics.NUM_STATES
        NUM_SAMPLES = self.TIME_EVOLUTION_STEPS
        self.ml_x0_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
    def _save_ml_onesample_dataset(self, t, new_x0, **kwargs):
        self.ml_x0_T[t] = new_x0


    def _create_data(self):
        '''原动力学模型产生的时间序列数据
        '''
        replication_data = self.run_replication()
        return replication_data

    def _create_trial_data(self):
        '''机器学习动力学模型产生的时间序列数据
        '''
        # 获取动力学数据
        self.logger.increase_indent()
        self.logger.log("Build machine learning time evolution dataset")
        self.network = self.network.get_data()  # 构造网络
        self.dynamics.set_network(self.network)  # 设置动力学网络
        self.dynamics.init_stateof_network()  # 在T_INIT时间后重置网络状态
        for t in range(self.TIME_EVOLUTION_STEPS):
            #  onestep_spread_result: 动力学的原本数据
            onestep_spread_result = self.dynamics._run_onestep()
            self.dynamics.spread_result_to_float(onestep_spread_result)
            # 模型预测结果：
            _, x0, y_pred, y_true, y_ob, w = self.ml_model.batch_task._do_batch_(self.ml_model.model, self.network, self.dynamics, tuple(onestep_spread_result.values()))
            new_x0 = self.dynamics.get_transition_state(y_pred.clone().detach())
            ml_onestep_spread_result = {
                "old_x0": x0.clone().detach(),
                "new_x0": new_x0.clone().detach(),
                "true_tp": y_true.clone().detach(),
                "weight": w.clone().detach()
            }
            # 将new_x0设置为当前状态
            # new_x0来自ml_onestep_spread_result：把机器学习的结果y_pred更新为当前动力学状态x0
            # new_x0来自onestep_spread_result：把动力学的new_x0更新为当前动力学状态x0
            self.dynamics.set_features(**ml_onestep_spread_result)
            # 保存结果
            self._save_ml_onesample_dataset(t, **ml_onestep_spread_result)
        self.data = {
            "data": self.ml_x0_T.sum(dim=1)/self.network.NUM_NODES

        }
        self.logger.decrease_indent()
        return self.data
