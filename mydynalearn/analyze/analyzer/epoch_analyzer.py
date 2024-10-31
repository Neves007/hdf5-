import os
import pandas as pd
import numpy as np

class epochAnalyzer():
    def __init__(self,config):
        self.config = config
        self.all_epoch_dataframe_file_path = self.__get_all_epoch_dataframe_file_path()
        self.best_epoch_dataframe_file_path = self.__get_best_epoch_dataframe()
        self.all_epoch_dataframe = self.__init_all_epoch_dataframe()
        self.best_epoch_dataframe = self.__init_best_epoch_dataframe()

    def __get_all_epoch_dataframe_file_path(self):
        root_dir_path = self.config.root_dir_path
        dataframe_dir_name = self.config.dataframe_dir_name
        # 创建文件夹
        dataframe_dir_path = os.path.join(root_dir_path, dataframe_dir_name)
        if not os.path.exists(dataframe_dir_path):
            os.makedirs(dataframe_dir_path)
        # 返回文件路径
        all_epoch_dataframe_file_name = self.config.all_epoch_dataframe_file_name
        all_epoch_dataframe_file_path = os.path.join(dataframe_dir_path, all_epoch_dataframe_file_name)
        return all_epoch_dataframe_file_path

    def __get_best_epoch_dataframe(self):
        root_dir_path = self.config.root_dir_path
        dataframe_dir_name = self.config.dataframe_dir_name
        # 创建文件夹
        dataframe_dir_path = os.path.join(root_dir_path, dataframe_dir_name)
        if not os.path.exists(dataframe_dir_path):
            os.makedirs(dataframe_dir_path)
        # 返回文件路径
        best_epoch_value_dataframe_file_name = self.config.best_epoch_value_dataframe_file_name
        best_epoch_value_dataframe_file_path = os.path.join(dataframe_dir_path, best_epoch_value_dataframe_file_name)
        return best_epoch_value_dataframe_file_path



    def __init_all_epoch_dataframe(self):
        all_epoch_dataframe = pd.DataFrame()
        return all_epoch_dataframe

    def __init_best_epoch_dataframe(self):
        best_epoch_dataframe = pd.DataFrame()
        return best_epoch_dataframe

    def save_all_epoch_dataframe(self):
        self.all_epoch_dataframe.to_csv(self.all_epoch_dataframe_file_path, index=False)

    def save_best_epoch_dataframe(self):
        self.best_epoch_dataframe.to_csv(self.best_epoch_dataframe_file_path, index=False)

    def load_all_epoch_dataframe(self):
        self.all_epoch_dataframe = pd.read_csv(self.all_epoch_dataframe_file_path)
    def load_best_epoch_dataframe(self):
        self.best_epoch_dataframe = pd.read_csv(self.best_epoch_dataframe_file_path)
    def analyze_result_to_pdseries(self,analyze_result):
        test_result_info = analyze_result['test_result_info']
        model_performance_dict = analyze_result['model_performance_dict']
        performance_dict = {"f1" : model_performance_dict["f1"],
                           "R" : model_performance_dict["R"],
                           "cross_loss" : model_performance_dict["cross_loss"],}
        analyze_result_series = pd.Series({**test_result_info,**performance_dict})
        return analyze_result_series

    def add_epoch_result(self, analyze_result):
        analyze_result_series = self.analyze_result_to_pdseries(analyze_result)
        self.all_epoch_dataframe = pd.concat([self.all_epoch_dataframe,analyze_result_series.to_frame().T])


    # 定义一个函数来判断R值是否稳定
    def find_best_epoch(self, sorted_group_data, threshold=0.0001, window_size=10):
        '''寻找使 R 值首次稳定的 model_exp_epoch_index

        :param r_values: 按照epoch排序的R值
        :param threshold: 判断稳定的阈值
        :param window_size: 窗口数
        :return: 使 R 值首次稳定的 epoch，始终不稳定返回-1
        '''
        # # 计算每个窗口的标准差
        # if window_size > len(r_values):
        #     raise ValueError("Window size cannot be larger than the list of r_values")
        #
        # # 遍历r_values找到符合条件的epoch
        # for i in range(len(r_values) - window_size):
        #     current_r = r_values[i]
        #     # 检查窗口中的R值是否都满足稳定性条件
        #     window_stable = all(abs(current_r - r) < threshold for r in r_values[i+1:i+window_size])
        #     if window_stable:
        #         return i  # 返回第一个稳定的epoch索引

        # 找到最小损失的索引
        loss_values = sorted_group_data['cross_loss'].values
        r_values = sorted_group_data['R'].values
        # 最佳epoch index
        best_epoch_index = np.argmin(loss_values)

        return best_epoch_index  # 如果没有找到符合条件的epoch，则返回None

    def get_best_epoch_index(self,
                             model_network_name,
                             model_dynamics_name,
                             dataset_network_name,
                             dataset_dynamics_name,
                             model_name,
                             **kwargs):
        """
        根据提供的参数在best_epoch_dataframe中查找对应的best_epoch和best_epoch。

        参数:
        - model_network: 模型网络
        - model_dynamics: 模型动力学
        - dataset_network: 数据集网络
        - dataset_dynamics: 数据集动力学
        - model: 模型标识符

        返回:
        - R_value: 稳定的R值
        - best_epoch: 达到最大R值的epoch
        """
        # 使用提供的参数在DataFrame中进行查找
        best_epoch_dataframe = self.best_epoch_dataframe
        query_result = best_epoch_dataframe[
            (best_epoch_dataframe['model_network_name'] == model_network_name) &
            (best_epoch_dataframe['model_dynamics_name'] == model_dynamics_name) &
            (best_epoch_dataframe['dataset_network_name'] == dataset_network_name) &
            (best_epoch_dataframe['dataset_dynamics_name'] == dataset_dynamics_name) &
            (best_epoch_dataframe['model_name'] == model_name)
            ]

        # 如果有符合条件的行，返回第一条记录的best_epoch和best_epoch
        if not query_result.empty:
            return query_result.iloc[0]['model_epoch_index']
        else:
            print("network: {}\ndynamics: {}\nmodel: {}\n ".format(model_network_name,model_dynamics_name,model_name,))
            # 如果没有找到符合条件的记录，可以返回None或者一些默认值或错误信息
            raise Exception('没有对应的值')

    def analyze_best_epoch(self):
        self.load_all_epoch_dataframe()
        if os.path.exists(self.best_epoch_dataframe_file_path):
            return
        else:
            group_cols = ["model_network_name",
                          "model_dynamics_name",
                          "dataset_network_name",
                          "dataset_dynamics_name",
                          "model_name"]

            # 通过分组获得使R值首次稳定的 model_exp_epoch_index
            grouped = self.all_epoch_dataframe.groupby(group_cols)

            # 遍历每个分组
            for group_name, group_data in grouped:
                # 获取当前分组的R值列表
                sorted_group_data = group_data.sort_values(by='model_epoch_index')
                # 遍历R值和epoch列表，找到最初进入稳定状态的epoch值
                best_epoch_index = self.find_best_epoch(sorted_group_data)
                best_epoch_series = group_data.iloc[best_epoch_index]

                self.best_epoch_dataframe = pd.concat([self.best_epoch_dataframe, best_epoch_series.to_frame().T])
            self.save_best_epoch_dataframe()