import itertools


class ParamsDealer:
    def __init__(self, params):
        self.params = params  # 内聚性增强：所有参数相关信息在初始化时加载一次

    def _generate_combinations(self, network_key, dynamics_key):
        """
        生成给定网络和动态类型的参数组合，并返回包含每一个配置的字典列表
        :param network_key: 网络类型键 (e.g., 'graph_network' or 'simplicial_network')
        :param dynamics_key: 动态类型键 (e.g., 'graph_dynamics' or 'simplicial_dynamics')
        :return: 包含每个配置字典的列表
        """
        if network_key in self.params and dynamics_key in self.params:
            # 动态构建需要排列的参数字典，确保键存在时才添加
            param_dict = {
                "network": self.params[network_key],
                "dynamics": self.params[dynamics_key]
            }
            # 将 self.params 中除网络和动态相关的其他参数加入 param_dict
            for key in self.params:
                if key not in ["graph_network", "graph_dynamics", "simplicial_network", "simplicial_dynamics"]:
                    param_dict[key] = self.params[key]

            # 使用 itertools.product 生成所有组合
            keys, values = zip(*param_dict.items())
            combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

            return combinations

        return []

    def get_graph_params(self):
        """获取图网络的参数组合"""
        params_graph = self._generate_combinations("graph_network", "graph_dynamics")
        return params_graph

    def get_simplex_params(self):
        """获取高阶网络的参数组合"""
        params_simplex = self._generate_combinations("simplicial_network", "simplicial_dynamics")
        return params_simplex

    def get_parse_params(self):
        """根据参数配置整合实验列表。

        :param only_higher_order: 是否只包含高阶网络。
        :return: 参数组合的列表。
        """
        params_graph = self.get_graph_params()
        params_simplex = self.get_simplex_params()

        # 根据标志选择返回高阶或全部组合
        return params_graph + params_simplex
