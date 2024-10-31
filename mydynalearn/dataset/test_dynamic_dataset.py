import os.path
import pickle
import torch
from tqdm import *
from abc import abstractmethod
from mydynalearn.networks import *
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class TestDynamicDataset():
    '''数据集类
    通过网络network和dynamics来说生成动力学数据集

    生成数据：
        - run_dynamic_process 连续时间动力学数据
        - run 生成动力学数据
    '''
    def __init__(self, config) -> None:
        self.config = config
        self.dataset_config = config.dataset
        self.NUM_SAMPLES = self.dataset_config.NUM_SAMPLES
        self.set_dataset_network()
        self.set_dataset_dynamics()
        self.DEVICE = self.config.DEVICE
        self.EFF_BETA_LIST = self.config.EFF_BETA_LIST
        self.stady_rho_dict = {key: 1.*torch.ones(len(self.EFF_BETA_LIST)) for key in self.dynamics.STATES_MAP.keys()}
        self.test_dynamics_file_path = self.get_test_dynamics_file_path()
        self.is_need_to_run = not os.path.exists(self.test_dynamics_file_path)

    def get_info(self):
        info = {
            'dynamic_name': self.dynamics.NAME,
            'network_name': self.network.NAME,
            'test_dynamics_file_path': self.test_dynamics_file_path
        }
        return info

    def show_info(self):
        info = self.get_info()
        print("test dynamics...\n network:{} \n dynamics: {}".format(info['network_name'],info['dynamic_name']))

    def save_dataset(self):
        file_name = self.test_dynamics_file_path
        with open(file_name, "wb") as file:
            pickle.dump(self,file)
        file.close()
    def load_dataset(self):
        file_name = self.test_dynamics_file_path
        with open(file_name, "rb") as file:
            data = pickle.load(file)
        file.close()
        return data
    def get_test_dynamics_file_path(self):
        dataset_dir_path = self.config.dataset_dir_path
        dynamic_name = self.dynamics.NAME
        network_name = self.network.NAME
        test_dynamics_file_name = "_".join([network_name,dynamic_name])+".pkl"
        test_dynamics_file_path = os.path.join(dataset_dir_path,test_dynamics_file_name)
        test_dynamics_file_path
        return test_dynamics_file_path

    def set_dataset_network(self):
        network = get_network(self.config)
        network.create_data()
        self.network = network

    def set_dataset_dynamics(self):
        dynamics = get_dynamics(self.config)
        dynamics.set_network(self.network)
        dynamics.init_stateof_network()
        self.dynamics = dynamics

    def init_dataset(self):
        assert self.network.MAX_DIMENSION == self.dynamics.MAX_DIMENSION
        NUM_NODES = self.network.NUM_NODES
        NUM_STATES = self.dynamics.NUM_STATES
        NUM_SAMPLES = self.NUM_SAMPLES
        self.x0_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.y_ob_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)
        self.y_true_T = torch.zeros(NUM_SAMPLES, NUM_NODES, NUM_STATES).to(self.config.DEVICE, dtype=torch.float)


    def save_onesample_dataset(self, t, old_x0, new_x0, true_tp, **kwargs):
        self.x0_T[t] = old_x0
        self.y_ob_T[t] = new_x0
        self.y_true_T[t] = true_tp

    def set_beta(self,eff_beta):
        self.dynamics.set_beta(eff_beta)


    def get_stady_rho(self):
        '''动力学实验
            简介
                - 运行一段连续时间演化的动力学过程
                - 用于验证动力学是否正确
        '''

        self.dynamics.init_stateof_network()
        print("create dynamics")
        for t in tqdm(range(self.NUM_SAMPLES)):
            onestep_spread_result = self.dynamics._run_onestep()
            self.dynamics.set_features(**onestep_spread_result)
            self.save_onesample_dataset(t, **onestep_spread_result)

        node_timeEvolution = self.y_ob_T
        fetch_final_node_states = node_timeEvolution[-100:]
        stady_rho = fetch_final_node_states.sum(dim=-2).mean(dim=0) / self.network.NUM_NODES
        return stady_rho


    def get_draw_data(self):
        data = {
            "stady_rho_dict": self.stady_rho_dict,
            "x": self.EFF_BETA_LIST,
        }
        return data

    def run(self):
        self.init_dataset()  # 设置
        for index, eff_beta in enumerate(self.EFF_BETA_LIST):
            self.set_beta(eff_beta)
            stady_rho_sum = 0
            for round_index in range(self.config.ROUND):
                stady_rho_temp = self.get_stady_rho()
                stady_rho_sum += stady_rho_temp
            stady_rho = stady_rho_sum/self.config.ROUND
            for STATES in self.dynamics.STATES_MAP:
                STATES_index = self.dynamics.STATES_MAP[STATES]
                self.stady_rho_dict[STATES][index] = stady_rho[STATES_index]
        self.save_dataset()

