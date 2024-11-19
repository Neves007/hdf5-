'''
Experiment_Compare_HighLow_AllModels：使用高阶网络、低阶网络和所有模型进行性能比较，探究不同模型的表现。
Experiment_SAT_HighOrder_LossK1K2：使用 SAT 模型对高阶网络的动力学进行实验，分析 loss-k1-k2 之间的关系。
Experiment_SAT_HighOrder_SCUAU_Attention：使用 SAT 模型对高阶玩具网络上的 SCUAU 动力学进行实验，观察 Attention Coefficients。
Experiment_SAT_HighOrder_Ablation：使用 SAT 模型对高阶网络动力学进行消融实验，通过调整参数探索不同配置的影响。
Experiment_SAT_vs_GNN_HighOrder：使用 SAT 模型和不同的 GNN 模型对高阶网络动力学进行实验，比较 SAT 模型与现有 GNN 模型在高阶动力学学习性能上的差异。
'''
# todo 更新参数存档点 1：修改包目录名为当前实验目录
from Experiment_Compare_HighLow_AllModels.params_dealer import ParamsDealer
from mydynalearn.experiments.experiment_manager import ExperimentManager

''' 所有参数
    "graph_network": ["ER", "SF"],
    "graph_dynamics": ["UAU", "CompUAU", "CoopUAU", "AsymUAU"],
    "simplicial_network": ["SCER", "SCSF"],
    "simplicial_real_network":["CONFERENCE", "HIGHSCHOOL", "HOSPITAL", "WORKPLACE"],
    "simplicial_dynamics": ["SCUAU", "SCCompUAU", "SCCoopUAU", "SCAsymUAU"],
    "graph_network": ["ER"],
    "graph_dynamics": ["UAU"],
    "simplicial_network": ["HIGHSCHOOL"],
    "simplicial_dynamics": ["SCUAU"],
    "model": ["GAT", "SAT"],  # 至少选一个
    "IS_WEIGHT": [False],
    "DEVICE": ['cuda'],
    "NUM_TEST": [1],
    "EPOCHS": [3],
    # 具体参数
    "NUM_SAMPLES": [100,500,1000,5000,10000],
    "T_INIT": [1,10,100,1000,10000],
    "SEED_FREC": [0.005,0.01,0.1,0.2,0.3],
'''
# todo 更新参数存档点 2：修改params_exp_dict的组成
params_exp_dict = {
    # 实验参数
    "graph_network": ["ER", "SF"],
    "graph_dynamics": ["UAU", "CompUAU", "CoopUAU", "AsymUAU"],
    "simplicial_network": ["SCER", "SCSF"],
    "simplicial_real_network":["CONFERENCE", "HIGHSCHOOL", "HOSPITAL", "WORKPLACE"],
    "simplicial_dynamics": ["SCUAU", "SCCompUAU", "SCCoopUAU", "SCAsymUAU"],
    "model": ["GAT", "SAT"],  # 至少选一个
    "IS_WEIGHT": [False],
    "DEVICE": ['cuda'],
    # 具体参数
    "NUM_TEST": [10],
    "EPOCHS": [30],
    "NUM_SAMPLES": [10000],
    "NUM_NODES":[1000]
}


if __name__ == '__main__':
    experiment_name = "Experiment_SAT_HighOrder_LossK1K2"
    params_dealer = ParamsDealer(params=params_exp_dict)
    params_exp = params_dealer.get_parse_params()  # 分解参数
    # 跑实验
    experiment_manager = ExperimentManager(params_exp)  # 返回实验对象
    experiment_manager.run(data_base = experiment_name)
