import os.path
import pickle
import torch
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics
from torch.utils.data import Dataset, Subset
from mydynalearn.logger import Log

class DynamicDatasetTimeEvolution(Dataset):
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
        self.need_to_run = not os.path.exists(self.dataset_file_path)
        self._init_dataset()


    def _get_dataset_file_path(self):
        dataset_file_name = f"TIME_EVOLUTION_DATASET_{self.network.NAME}_{self.dynamics.NAME}_{self.exp.model.NAME}_epoch{self.ml_model.epoch_index}.pkl"
        return os.path.join(self.config.dataset_dir_path, dataset_file_name)

    def __len__(self) -> int:
        return 1
    def __getitem__(self, index):
        x0 = self.cur_x0_for_ml
        y_ob = self.cur_y_ob_for_ml
        y_true = self.cur_y_true_for_ml
        weight = self.cur_weight_for_ml
        return x0, y_ob, y_true, weight

    def get_time_evolution_dataset(self):
        return Subset(self, list(range(self.TIME_EVOLUTION_STEPS)))
    def _get_dataset_network(self):
        network = get_network(self.config)
        return network

    def _get_dataset_dynamics(self):
        dynamics = get_dynamics(self.config)
        return dynamics
    def is_dataset_exist(self):
        return os.path.exists(self.dataset_file_path)

    def _init_dataset(self):
        assert self.network.MAX_DIMENSION == self.dynamics.MAX_DIMENSION
        NUM_NODES = self.network.NUM_NODES
        NUM_STATES = self.dynamics.NUM_STATES
        NUM_SAMPLES = self.TIME_EVOLUTION_STEPS
        self.ori_x0_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.ori_y_ob_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.ori_y_true_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.ori_weight_T = torch.zeros(NUM_SAMPLES, NUM_NODES).to(self.config.DEVICE, dtype=torch.float)
        self.ml_x0_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.ml_y_ob_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.ml_y_true_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.ml_weight_T = torch.zeros(NUM_SAMPLES, NUM_NODES).to(self.config.DEVICE, dtype=torch.float)


    def _save_ori_onesample_dataset(self, t, old_x0, new_x0, true_tp, weight, **kwargs):
        self.ori_x0_T[t] = old_x0
        self.ori_y_ob_T[t] = new_x0
        self.ori_y_true_T[t] = true_tp
        self.ori_weight_T[t] = weight

    def _save_ml_onesample_dataset(self, t, old_x0, new_x0, true_tp, weight, **kwargs):
        self.ml_x0_T[t] = old_x0
        self.ml_y_ob_T[t] = new_x0
        self.ml_y_true_T[t] = true_tp
        self.ml_weight_T[t] = weight

    def _save_dataset(self):
        self.logger.increase_indent()
        self.logger.log(f"save: {self.dataset_file_path}")
        data = {
            "ori_x0_T": self.ori_x0_T,
            "ml_x0_T": self.ml_x0_T,
        }
        file_name = self.dataset_file_path
        with open(file_name, "wb") as file:
            pickle.dump(data, file)
        file.close()
        self.logger.decrease_indent()

    def _load_dataset(self):
        self.logger.increase_indent()
        file_name = self.dataset_file_path
        self.logger.log(f"load: {file_name}")
        with open(file_name, "rb") as file:
            data = pickle.load(file)
        file.close()
        self.logger.decrease_indent()
        return data

    def get_dataset(self):
        data = {
            "ori_x0_T": self.ori_x0_T,
            "ml_x0_T": self.ml_x0_T,
        }
        return data

    def _buid_dataset_original_time_evolution(self):
        '''原动力学模型产生的时间序列数据
        '''
        # 获取动力学数据
        self.network.create_data()  # 构造网络
        self.dynamics.set_network(self.network)  # 设置动力学网络
        self.dynamics.init_stateof_network()  # 在T_INIT时间后重置网络状态
        for t in range(self.TIME_EVOLUTION_STEPS):
            onestep_spread_result = self.dynamics._run_onestep()
            self.dynamics.set_features(**onestep_spread_result)
            self._save_ori_onesample_dataset(t, **onestep_spread_result)

    def set_cur_state(self, old_x0, new_x0, true_tp, weight):
        self.cur_x0_for_ml = old_x0
        self.cur_y_ob_for_ml = new_x0
        self.cur_y_true_for_ml = true_tp
        self.cur_weight_for_ml = weight
    def _buid_dataset_ml_time_evolution(self):
        '''机器学习动力学模型产生的时间序列数据
        '''
        # 获取动力学数据
        self.logger.increase_indent()
        self.logger.log("Build machine learning time evolution dataset")
        self.network.create_data()  # 构造网络
        self.dynamics.set_network(self.network)  # 设置动力学网络
        self.dynamics.init_stateof_network()  # 在T_INIT时间后重置网络状态
        for t in range(self.TIME_EVOLUTION_STEPS):
            onestep_spread_result = self.dynamics._run_onestep()
            self.dynamics.spread_result_to_float(onestep_spread_result)
            _, x0, y_pred, y_true, y_ob, w = self.ml_model.batch_task._do_batch_(self.ml_model.model, self.network, self.dynamics, tuple(onestep_spread_result.values()))
            new_x0 = self.dynamics.get_transition_state(y_pred.clone().detach())
            ml_onestep_spread_result = {
                "old_x0": x0.clone().detach(),
                "new_x0": new_x0.clone().detach(),
                "true_tp": y_true.clone().detach(),
                "weight": w.clone().detach()
            }
            self.dynamics.set_features(**ml_onestep_spread_result)
            self._save_ml_onesample_dataset(t, **ml_onestep_spread_result)
        self.logger.decrease_indent()

    def _build_and_save_new_dataset(self):
        """
        构建新的数据集并保存到文件中。
        """
        # 构建原始动力学模型的数据集
        self.logger.increase_indent()
        self.logger.log("build and save time evolution dataset")
        self._buid_dataset_original_time_evolution()
        self._buid_dataset_ml_time_evolution()
        self._save_dataset()
        self.logger.decrease_indent()
        return self.get_dataset()

    def run(self):
        """
        加载或构建动力学数据集，并返回网络、动力学和时间演化数据集。
        """
        self.logger.increase_indent()
        self.logger.log("run time evolution dynamics dataset")
        if self.is_dataset_exist():
            dataset = self._load_dataset()
            # todo: 修改名称错误
        else:
            dataset = self._build_and_save_new_dataset()
        self.logger.decrease_indent()
        return dataset