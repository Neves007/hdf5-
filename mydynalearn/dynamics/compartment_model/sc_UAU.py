import copy
import torch
import random
from mydynalearn.dynamics.compartment_model import CompartmentModel
from ..simple_dynamic_weight.simple_dynamic_weight import SimpleDynamicWeight
#  进行一步动力学
class SCUAU(CompartmentModel):
    def __init__(self,config):
        super().__init__(config)
        self.EFF_AWARE = torch.tensor(self.dynamics_config.EFF_AWARE)
        self.EFF_AWARE_DELTA = torch.tensor(self.dynamics_config.EFF_AWARE_DELTA)
        self.MU = self.dynamics_config.MU
        self.SEED_FREC = self.dynamics_config.SEED_FREC
    def set_beta(self,eff_beta):
        self.EFF_AWARE = eff_beta
    def _init_x0(self):
        x0 = torch.zeros(self.NUM_NODES).to(self.DEVICE,torch.long)
        x0 = self.NODE_FEATURE_MAP[x0]

        NUM_SEED_NODES = int(self.NUM_NODES * self.SEED_FREC)
        AWARE_SEED_INDEX = random.sample(range(self.NUM_NODES), NUM_SEED_NODES)
        x0[AWARE_SEED_INDEX] = self.NODE_FEATURE_MAP[self.STATES_MAP["A"]]
        self.x0=x0
    def get_adj_activate_simplex(self):
        '''
        聚合邻居单纯形信息
        了解节点周围单纯形，【非激活态，激活态】数量
        '''
        # inc_matrix_adj_act_edge：（节点数，边数）表示节点i与边j，相邻且j是激活边
        inc_matrix_adj_act_edge = self.get_inc_matrix_adjacency_activation(inc_matrix_col_feature=self.x1,
                                                      _threshold_scAct=1,
                                                      target_state='A',
                                                      inc_matrix_adj=self.network.inc_matrix_adj1)
        inc_matrix_adj_act_triangle = self.get_inc_matrix_adjacency_activation(inc_matrix_col_feature=self.x2,
                                                      _threshold_scAct=2,
                                                      target_state='A',
                                                      inc_matrix_adj=self.network.inc_matrix_adj2)

        # 检查4号节点的情况
        # target_node = 4
        # act_tri_index = torch.where(inc_matrix_adj_act_triangle.to_dense()[target_node]==1)[0]
        # act_tri = self.network.triangles[act_tri_index]
        # act_tri_state = self.x2[act_tri_index]
        #
        # act_edge_index = torch.where(inc_matrix_adj_act_edge.to_dense()[target_node]==1)[0]
        # act_edge = self.network.edges[act_edge_index]
        # act_edge_state = self.x1[act_edge_index]

        # adj_act_edges：（节点数）表示节点i相邻激活边数量
        adj_act_edges = torch.sparse.sum(inc_matrix_adj_act_edge,dim=1).to_dense()
        adj_act_triangles = torch.sparse.sum(inc_matrix_adj_act_triangle,dim=1).to_dense()

        return adj_act_edges,adj_act_triangles

    def _preparing_spreading_data(self):
        adj_act_edges,adj_act_triangles = self.get_adj_activate_simplex()
        old_x0 = copy.deepcopy(self.x0)
        old_x1 = copy.deepcopy(self.x1)
        old_x2 = copy.deepcopy(self.x2)
        true_tp = torch.zeros(self.x0.shape).to(self.DEVICE)
        return old_x0, old_x1, old_x2, true_tp, adj_act_edges, adj_act_triangles

    def _get_nodeid_for_each_state(self):
        U_index = torch.where(self.x0[:, self.STATES_MAP["U"]] == 1)[0].to(self.DEVICE, dtype=torch.long)
        A_index = torch.where(self.x0[:, self.STATES_MAP["A"]] == 1)[0].to(self.DEVICE, dtype=torch.long)
        return U_index, A_index

    def _get_new_feature(self, x0, inf_U_index, recover_A_index):
        if inf_U_index.shape[0] > 0:
            x0[inf_U_index, :] = self.NODE_FEATURE_MAP[1]  # S->I
        if recover_A_index.shape[0] > 0:
            x0[recover_A_index, :] = self.NODE_FEATURE_MAP[0]  # I->S
        x1 = self.get_x1_from_x0(x0, self.network)
        x2 = self.get_x2_from_x0(x0, self.network)
        return x0, x1, x2

    def _dynamic_for_node_A(self, A_index, true_tp):
        true_tp[A_index, self.STATES_MAP["U"]] = self.MU
        true_tp[A_index, self.STATES_MAP["A"]] = 1 - self.MU

    def _dynamic_for_node_U(self, U_index, adj_act_edges, adj_act_triangles, true_tp):
        # 感染概率
        not_aware_prob = torch.pow(1 - self.BETA, adj_act_edges)
        not_aware_prob_DELTA = torch.pow(1 - self.BETA_DELTA, adj_act_triangles)
        # 不被感染的概率矩阵
        not_aware_prob = not_aware_prob * not_aware_prob_DELTA
        # 感染的概率 aware_prob (infected probability)
        aware_prob = 1 - not_aware_prob

        # 修改实际迁移概率
        true_tp[U_index, self.STATES_MAP["U"]] = 1 - aware_prob[U_index]
        true_tp[U_index, self.STATES_MAP["A"]] = aware_prob[U_index]

    def _spread(self):
        old_x0, old_x1, old_x2, true_tp, adj_act_edges, adj_act_triangles = self._preparing_spreading_data()
        U_index, A_index = self._get_nodeid_for_each_state()
        self._dynamic_for_node_U(U_index, adj_act_edges, adj_act_triangles, true_tp)
        self._dynamic_for_node_A(A_index, true_tp)
        new_x0 = self.get_transition_state(true_tp)
        weight = 1. * torch.ones(self.NUM_NODES).to(self.DEVICE)
        spread_result = {
            "old_x0":old_x0,
            "new_x0":new_x0,
            "true_tp":true_tp,
            "weight": weight
        }
        return spread_result
    def _run_onestep(self):
        self.BETA = self.EFF_AWARE * self.MU / self.network.AVG_K
        self.BETA_DELTA = self.EFF_AWARE_DELTA * self.MU / self.network.AVG_K_DELTA
        spread_result = self._spread()
        return spread_result

    def get_adj_activate_simplex_dict(self):
        '''
        将adj_activate_simplex聚合dict返回
        :return:
        '''
        adj_act_edges,adj_act_triangles = self.get_adj_activate_simplex()
        adj_activate_simplex_dict = {
            "adj_act_edges": adj_act_edges.to('cpu').numpy(),
            "adj_act_triangles": adj_act_triangles.to('cpu').numpy(),
        }

        return adj_activate_simplex_dict