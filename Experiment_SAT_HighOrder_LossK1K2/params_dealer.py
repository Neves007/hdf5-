import itertools


class ParamsDealer:
    def __init__(self, params):
        self.params = params  # 内聚性增强：所有参数相关信息在初始化时加载一次

    def _generate_combinations(self, param_dict):

        # 使用 itertools.product 生成所有组合
        keys, values = zip(*param_dict.items())
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        return combinations

    def get_graph_params(self):
        """获取图网络的参数组合"""
        if "graph_network" in self.params.keys() and "graph_dynamics" in self.params.keys():
            param_dict = {
                "NETWORK_NAME": self.params["simplicial_network"],
                "DYNAMIC_NAME": self.params["simplicial_dynamics"],
                "MODEL_NAME": self.params["model"],
                "IS_WEIGHT": self.params["IS_WEIGHT"],
                "DEVICE": self.params["DEVICE"],
                "NUM_TEST": self.params["NUM_TEST"],
                "EPOCHS": self.params['EPOCHS'],
                "NUM_NODES": self.params["NUM_NODES"],
                "NUM_SAMPLES": self.params["NUM_SAMPLES"]
            }
            params_graph = self._generate_combinations(param_dict)
        else:
            params_graph = []
        return params_graph

    def get_simplex_params(self):
        """获取高阶网络的参数组合"""
        if "simplicial_network" in self.params.keys() and "simplicial_dynamics" in self.params.keys():
            param_dict = {
                "NETWORK_NAME": self.params["simplicial_network"],
                "DYNAMIC_NAME": self.params["simplicial_dynamics"],
                "MODEL_NAME": self.params["model"],
                "IS_WEIGHT": self.params["IS_WEIGHT"],
                "DEVICE": self.params["DEVICE"],
                "NUM_TEST": self.params["NUM_TEST"],
                "EPOCHS": self.params['EPOCHS'],
                "NUM_NODES": self.params["NUM_NODES"],
                "NUM_SAMPLES": self.params["NUM_SAMPLES"]
            }
            params_simplex = self._generate_combinations(param_dict)
        else:
            params_simplex = []
        return params_simplex

    def get_real_simplex_params(self):
        """获取高阶网络的参数组合"""
        if "simplicial_real_network" in self.params.keys() and "simplicial_dynamics" in self.params.keys():
            param_dict = {
                "NETWORK_NAME": self.params["simplicial_real_network"],
                "DYNAMIC_NAME": self.params["simplicial_dynamics"],
                "MODEL_NAME": self.params["model"],
                "IS_WEIGHT": self.params["IS_WEIGHT"],
                "DEVICE": self.params["DEVICE"],
                "NUM_TEST": self.params["NUM_TEST"],
                "EPOCHS": self.params['EPOCHS'],
                "NUM_SAMPLES": self.params["NUM_SAMPLES"]
            }
            params_real_simplex = self._generate_combinations(param_dict)
        else:
            params_real_simplex = []
        return params_real_simplex

    def get_parse_params(self):
        """根据参数配置整合实验列表。

        :param only_higher_order: 是否只包含高阶网络。
        :return: 参数组合的列表。
        """
        params_graph = self.get_graph_params()
        params_simplex = self.get_simplex_params()
        params_real_simplex = self.get_real_simplex_params()
        # todo 更新参数存档点 3：修改parse_params的组成
        # 根据标志选择返回高阶或全部组合
        return params_simplex
