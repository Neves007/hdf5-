import itertools


class ParamsDealer:
    def __init__(self, params):
        self.params = params  # 内聚性增强：所有参数相关信息在初始化时加载一次

    def _generate_combinations(self, network_key, dynamics_key):
        """生成给定网络和动态类型的参数组合，动态检查其他键的存在性并决定是否加入组合"""
        if network_key in self.params and dynamics_key in self.params:
            # 动态构建需要排列的参数列表，确保键存在时才添加
            param_list = [
                self.params[network_key],
                self.params[dynamics_key]
            ]
            if "model" in self.params:
                param_list.append(self.params["model"])
            if "IS_WEIGHT" in self.params:
                param_list.append(self.params["IS_WEIGHT"])

            return list(itertools.product(*param_list))

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
