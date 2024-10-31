import itertools
class PasramsDealer:
    def __init__(self,params):
        params = params
        
    @staticmethod
    def assemble_train_params(params, only_higher_order=False):
        ''' 通过给定的参数整合实验列表

        :param only_higher_order: 是否只包含高阶网络
        :return:
        '''
        train_params_graph = []
        train_params_simplex = []

        if "grpah_network" in params:
            assert "grpah_dynamics" in params
            train_params_graph = list(
                itertools.product(
                    params["grpah_network"],
                    params["grpah_dynamics"],
                    params["model"],
                    params["IS_WEIGHT"]))
        if "simplicial_network" in params:
            assert "simplicial_dynamics" in params
            train_params_simplex = list(
                itertools.product(
                    params["simplicial_network"],
                    params["simplicial_dynamics"],
                    params["model"],
                    params["IS_WEIGHT"]))
        if only_higher_order:
            train_params = train_params_simplex
        else:
            train_params = train_params_graph + train_params_simplex
        return train_params
    @staticmethod
    def assemble_test_dynamics_params(params, only_higher_order=False):
        ''' 通过给定的参数整合实验列表

        :param only_higher_order: 是否只包含高阶网络
        :return:
        '''
        train_params_graph = []
        train_params_simplex = []

        if "grpah_network" in params:
            assert "grpah_dynamics" in params
            train_params_graph = list(
                itertools.product(
                    params["grpah_network"],
                    params["grpah_dynamics"]))
        if "simplicial_network" in params:
            assert "simplicial_dynamics" in params
            train_params_simplex = list(
                itertools.product(
                    params["simplicial_network"],
                    params["simplicial_dynamics"]))
        if only_higher_order:
            train_params = train_params_simplex
        else:
            train_params = train_params_graph + train_params_simplex
        return train_params
