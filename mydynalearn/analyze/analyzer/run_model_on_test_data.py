import os
import pickle
import torch
import numpy as np
from mydynalearn.analyze.utils.data_handler.dynamic_data_handler import DynamicDataHandler
from multipledispatch import dispatch

class runModelOnTestData:
    def __init__(self, config, model_exp, dataset_exp):
        """
        测试类，使用 testdata_exp 的测试数据来测试 model_exp 的模型结果
        :param config: 配置对象
        :param model_exp: 模型实验对象
        :param dataset_exp: 数据集实验对象
        """
        self.config = config
        self.IS_WEIGHT = model_exp.config.IS_WEIGHT
        self.model_exp = model_exp
        self.testdata_exp = dataset_exp
        self.EPOCHS = model_exp.model.config.model.EPOCHS
        self.exp_info = self._get_exp_info()
        self.all_need_to_run = self.check_run_necessity()

    def _get_exp_info(self):
        """
        获取全局信息，包括模型和数据集的相关配置
        """
        return {
            "model_network_name": self.model_exp.config.network.NAME,
            "model_dynamics_name": self.model_exp.config.dynamics.NAME,
            "dataset_network_name": self.testdata_exp.config.network.NAME,
            "dataset_dynamics_name": self.testdata_exp.config.dynamics.NAME,
            "model_name": self.model_exp.config.model.NAME,
        }

    @dispatch()
    def check_run_necessity(self):
        """
        检查所有 epochs 是否需要运行分析
        """
        return any(self.check_run_necessity(epoch) for epoch in range(self.EPOCHS))

    @dispatch(int)
    def check_run_necessity(self, model_exp_epoch_index):
        """
        检查特定 epoch 是否需要运行分析
        """
        analyze_result_filepath = self.get_analyze_result_filepath(model_exp_epoch_index)
        return not os.path.exists(analyze_result_filepath)

    def get_analyze_result_filepath(self, model_exp_epoch_index):
        """
        构建分析结果的文件路径
        """
        model_info = f"{self.exp_info['model_network_name']}_{self.exp_info['model_dynamics_name']}_{self.exp_info['model_name']}"
        testdata_info = f"{self.exp_info['dataset_network_name']}_{self.exp_info['dataset_dynamics_name']}_{self.exp_info['model_name']}"
        model_dir_name = f"model_{model_info}"
        testdata_dir_name = f"testdata_{testdata_info}"

        analyze_result_dir_path = os.path.join(
            self.config.root_dir_path,
            self.config.analyze_result_dir_name,
            model_dir_name,
            testdata_dir_name
        )
        os.makedirs(analyze_result_dir_path, exist_ok=True)
        return os.path.join(analyze_result_dir_path, f"epoch{model_exp_epoch_index}_analyze_result.pkl")

    def save_analyze_result(self, analyze_result, analyze_result_filepath):
        """
        保存分析结果
        """
        with open(analyze_result_filepath, "wb") as f:
            pickle.dump(analyze_result, f)

    def load_analyze_result(self, analyze_result_filepath):
        """
        加载分析结果
        """
        with open(analyze_result_filepath, "rb") as f:
            return pickle.load(f)

    def compute_R(self, test_result_curepoch):
        """
        计算相关系数 R
        """
        y_pred = torch.cat([data["y_pred"] for data in test_result_curepoch], dim=0)
        y_true = torch.cat([data["y_true"] for data in test_result_curepoch], dim=0)
        y_ob = torch.cat([data["y_ob"] for data in test_result_curepoch], dim=0)

        R_input_y_pred = y_pred[torch.where(y_ob == 1)].detach().numpy()
        R_input_y_true = y_true[torch.where(y_ob == 1)].detach().numpy()
        return np.corrcoef(R_input_y_pred, R_input_y_true)[0, 1]

    def compute_loss(self, test_result_curepoch):
        """
        计算损失
        """
        loss_list = torch.stack([data["loss"] for data in test_result_curepoch])
        return loss_list.mean().detach().item()

    def create_analyze_result(self, model_exp_epoch_index):
        """
        创建分析结果
        """
        try:
            network, dynamics, _, _, test_loader = self.testdata_exp.dataset.load()
        except Exception as e:
            print(self.exp_info)
            raise e

        test_result_time_list = self.model_exp.model.epoch_tasks.run_test_epoch(
            network, dynamics, test_loader, model_exp_epoch_index
        )

        dynamic_data_handler = DynamicDataHandler(dynamics, test_result_time_list)
        test_result_info = {
            "model_network_name": self.exp_info["model_network_name"],
            "model_dynamics_name": self.exp_info["model_dynamics_name"],
            "MAX_DIMENSION": dynamics.MAX_DIMENSION,
            "dataset_network_name": self.exp_info["dataset_network_name"],
            "dataset_dynamics_name": self.exp_info["dataset_dynamics_name"],
            "model_name": self.exp_info["model_name"],
            "model_epoch_index": model_exp_epoch_index,
        }

        test_result_df = dynamic_data_handler.get_result_dataframe(test_result_info)
        model_performance_dict = dynamic_data_handler.get_model_performance(test_result_df)
        return {
            "test_result_info": test_result_info,
            "test_result_df": test_result_df,
            "network": network,
            "dynamics": dynamics,
            "dynamics_STATES_MAP": dynamics.STATES_MAP,
            "model_performance_dict": model_performance_dict,
        }



    def _get_or_create_analyze_result(self, analyze_result_filepath, model_exp_epoch_index):
        """
        获取或创建分析结果
        """
        if not os.path.exists(analyze_result_filepath):
            analyze_result = self.create_analyze_result(model_exp_epoch_index)
            self.save_analyze_result(analyze_result, analyze_result_filepath)
        else:
            analyze_result = self.load_analyze_result(analyze_result_filepath)
        return analyze_result

    def _print_analyze_summary(self, model_exp_epoch_index, analyze_result_filepath):
        """
        打印分析摘要
        """
        print("Testing:")
        print(f"Analyze epoch: {model_exp_epoch_index}")
        print(f"Output analyze_result_filepath: {analyze_result_filepath}")
        print("Analyze completed!")

    def run(self, model_exp_epoch_index):
        """
        运行指定 epoch 的分析
        """
        analyze_result_filepath = self.get_analyze_result_filepath(model_exp_epoch_index)
        analyze_result = self._get_or_create_analyze_result(analyze_result_filepath, model_exp_epoch_index)
        self._print_analyze_summary(model_exp_epoch_index, analyze_result_filepath)
        return analyze_result

