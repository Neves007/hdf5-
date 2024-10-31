import copy
import torch
import random
from mydynalearn.dynamics.compartment_model import CompartmentModel
from ..simple_dynamic_weight.simple_dynamic_weight import SimpleDynamicWeight
class SCCoevUAU(CompartmentModel):
    '''Markdown
     协同动力学
    节点状态：UU,A1U,UA2,A1A2
    动力学参数:

        BETA_A1: 被A1邻居影响的概率
        BETA_A2: 被A2邻居影响的概率
        BETA_DELTA_A1: 被A1高阶邻居影响的概率
        BETA_DELTA_A2: 被A2高阶邻居影响的概率
        prime_A1: [0,+inf] 控制协同传播的参数
        prime_A2: [0,+inf] 控制协同传播的参数
        BETA_PRIME_A1 = 1 − (1 − BETA_A1)^prime_A1:
        BETA_PRIME_A2 = 1 − (1 − BETA_A2)^prime_A2:
        BETA_DELTA_PRIME_A1 = 1 − (1 − BETA_DELTA_A1)^prime_A1:
        BETA_DELTA_PRIME_A2 = 1 − (1 − BETA_DELTA_A2)^prime_A2:
        MU: 恢复概率
        N_A1U: A1U态邻居数
        N_UA2: UA2态邻居数
        N_A1A2: A1A2态邻居数
        N_A1U_DELTA: A1U态二阶邻居数
        N_UA2_DELTA: UA2态二阶邻居数
        N_A1A2_DELTA: A1A2态二阶邻居数
        
        # 一阶感染
        q_A1U = (1 - BETA_A1)^N_A1U: 不被A1U邻居影响的概率
        q_UA2 = (1 - BETA_A2)^N_UA2: 不被UA2邻居影响的概率
        A1A2 传播 A1的概率:
            BETA_A1A2ToA1 = BETA_A1(1-BETA_A2)/ (BETA_A1(1-BETA_A2) + BETA_A2(1-BETA_A1))
        A1A2 传播 A2的概率:
            BETA_A1A2ToA2 = BETA_A2(1-BETA_A1)/ (BETA_A1(1-BETA_A2) + BETA_A2(1-BETA_A1))
        q_A1A2ToA1 = (1 - BETA_A1A2ToA1)^N_A1A2
        q_A1A2ToA2 = (1 - BETA_A1A2ToA2)^N_A1A2
        
        # 二阶感染 
        q_DELTA_A1U = (1 - BETA_DELTA_A1)^N_DELTA_A1U: 不被A1U邻居影响的概率
        q_DELTA_UA2 = (1 - BETA_DELTA_A2)^N_DELTA_UA2: 不被UA2邻居影响的概率
        A1A2 传播 A1的概率:
            BETA_DELTA_A1A2ToA1 = BETA_DELTA_A1(1-BETA_DELTA_A2)/ (BETA_DELTA_A1(1-BETA_DELTA_A2) + BETA_DELTA_A2(1-BETA_DELTA_A1))
        A1A2 传播 A2的概率:
            BETA_DELTA_A1A2ToA2 = BETA_DELTA_A2(1-BETA_DELTA_A1)/ (BETA_DELTA_A1(1-BETA_DELTA_A2) + BETA_DELTA_A2(1-BETA_DELTA_A1))
        q_DELTA_A1A2ToA1 = (1 - BETA_DELTA_A1A2ToA1)^N_DELTA_A1A2
        q_DELTA_A1A2ToA2 = (1 - BETA_DELTA_A1A2ToA2)^N_DELTA_A1A2
        
        
        # 一阶促进感染
        q_PRIME_A1 = (1 - BETA_PRIME_A1)^(N_A1U+N_A1A2): 不被A1U邻居影响的概率
        q_PRIME_A2 = (1 - BETA_PRIME_A2)^(N_UA2+N_A1A2): 不被UA2邻居影响的概率
        
        # 二阶促进感染
        q_DELTA_PRIME_A1 = (1 - BETA_DELTA_PRIME_A1)^(N_A1U+N_A1A2): 不被A1U邻居影响的概率
        q_DELTA_PRIME_A2 = (1 - BETA_DELTA_PRIME_A2)^(N_UA2+N_A1A2): 不被UA2邻居影响的概率
        
        # UU态传播A1还是A2
        q_A1 = q_A1U * q_A1A2ToA1 * q_DELTA_A1U * q_DELTA_A1A2ToA1
        q_A2 = q_UA2 * q_A1A2ToA2 * q_DELTA_UA2 * q_DELTA_A1A2ToA2

        g_A1 = 1 - q_A1: 与所有A1态邻居接触，且被影响成为A1态的概率
        g_A2 = 1 - q_A2: 与所有A2态邻居接触，且被影响成为A2态的概率


        f_A1 = g_A1*(1-g_A2)/(g_A1*(1-g_A2)+g_A2*(1-g_A1)): 当已知节点被影响，被影响成为A1态的概率
        f_A2 = g_A2*(1-g_A1)/(g_A1*(1-g_A2)+g_A2*(1-g_A1)): 当已知节点被影响，被影响成为A2态的概率


    传播模型：
        UU -> UU:
            q_A1 * q_A2
        UU -> A1U:
            (1 - q_A1 * q_A2) * f_A1
        UU -> UA2:
            (1 - q_A1 * q_A2) * f_A2
        UU -> A1A2:
            0

        A1U -> UU:
            MU * q_PRIME_A2 * q_DELTA_PRIME_A2
        A1U -> A1U:
            (1 - MU) * q_PRIME_A2 * q_DELTA_PRIME_A2
        A1U -> UA2:
            Mu * (1 - q_PRIME_A2 * q_DELTA_PRIME_A2)
        A1U -> A1A2:
            (1 - MU) * (1 - q_PRIME_A2 * q_DELTA_PRIME_A2)

        UA2 -> UU:
            q_PRIME_A1 * q_DELTA_PRIME_A1* MU
        UA2 -> A1U:
            (1 - q_PRIME_A1 * q_DELTA_PRIME_A1) * Mu
        UA2 -> UA2:
            q_PRIME_A1 * q_DELTA_PRIME_A1 * (1 - MU)
        UA2 -> A1A2:
            (1 - q_PRIME_A1 * q_DELTA_PRIME_A1) * (1 - MU)

        recovery_prob:
            1 - (1 - self.MU_A1) * (1 - self.MU_A2)
        f_MU_A1:
            MU_A1 * (1 - MU_A2) / (MU_A2 * (1 - MU_A1) + MU_A1 * (1 - MU_A2))
        f_MU_A2:
            MU_A2 * (1 - MU_A1) / (MU_A2 * (1 - MU_A1) + MU_A1 * (1 - MU_A2))
        A1A2 -> UU:
            0
        A1A2 -> A1U:
            recovery_prob * f_MU_A1
        A1A2 -> UA2:
            recovery_prob * f_MU_A2
        A1A2 -> A1A2:
            1 - recovery_prob

    '''
    def __init__(self, config):
        super().__init__(config)
        self.EFF_AWARE_A1 = torch.tensor(self.dynamics_config.EFF_AWARE_A1)
        self.EFF_AWARE_A2 = torch.tensor(self.dynamics_config.EFF_AWARE_A2)
        self.EFF_AWARE_DELTA_A1 = torch.tensor(self.dynamics_config.EFF_AWARE_DELTA_A1)
        self.EFF_AWARE_DELTA_A2 = torch.tensor(self.dynamics_config.EFF_AWARE_DELTA_A2)
        self.prime_A1 = torch.tensor(self.dynamics_config.prime_A1)
        self.prime_A2 = torch.tensor(self.dynamics_config.prime_A2)
        self.MU_A1 = self.dynamics_config.MU_A1
        self.MU_A2 = self.dynamics_config.MU_A2
        self.SEED_FREC_A1 = self.dynamics_config.SEED_FREC_A1
        self.SEED_FREC_A2 = self.dynamics_config.SEED_FREC_A2


    def set_beta(self,eff_beta):
        self.EFF_AWARE_A1 = eff_beta

    def _init_x0(self):
        x0 = torch.zeros(self.NUM_NODES).to(self.DEVICE,torch.long)
        x0 = self.NODE_FEATURE_MAP[x0]
        NUM_SEED_NODES_A1 = int(self.NUM_NODES * self.SEED_FREC_A1)
        NUM_SEED_NODES_A2 = int(self.NUM_NODES * self.SEED_FREC_A2)
        AWARE_SEED_INDEX_all = random.sample(range(self.NUM_NODES), NUM_SEED_NODES_A1+NUM_SEED_NODES_A2)
        AWARE_SEED_INDEX_A1 = AWARE_SEED_INDEX_all[:NUM_SEED_NODES_A1]
        AWARE_SEED_INDEX_A2 = AWARE_SEED_INDEX_all[NUM_SEED_NODES_A1:]
        x0[AWARE_SEED_INDEX_A1] = self.NODE_FEATURE_MAP[self.STATES_MAP["A1U"]]
        x0[AWARE_SEED_INDEX_A2] = self.NODE_FEATURE_MAP[self.STATES_MAP["UA2"]]
        self.x0=x0

    def get_adj_activate_simplex(self):
        '''
        聚合邻居单纯形信息
        了解节点周围单纯形，【非激活态，激活态】数量
        '''
        # inc_matrix_adj_act_edge：（节点数，边数）表示节点i与边j，相邻且j是激活边
        inc_matrix_adj_A1U_act_edge = self.get_inc_matrix_adjacency_activation(inc_matrix_col_feature=self.x1,
                                                                              _threshold_scAct=1,
                                                                              target_state='A1U',
                                                                              inc_matrix_adj=self.network.inc_matrix_adj1)
        inc_matrix_adj_UA2_act_edge = self.get_inc_matrix_adjacency_activation(inc_matrix_col_feature=self.x1,
                                                                              _threshold_scAct=1,
                                                                              target_state='UA2',
                                                                              inc_matrix_adj=self.network.inc_matrix_adj1)
        inc_matrix_adj_A1A2_act_edge = self.get_inc_matrix_adjacency_activation(inc_matrix_col_feature=self.x1,
                                                                              _threshold_scAct=1,
                                                                              target_state='A1A2',
                                                                              inc_matrix_adj=self.network.inc_matrix_adj1)
        inc_matrix_adj_A1U_act_triangle = self.get_inc_matrix_adjacency_activation(inc_matrix_col_feature=self.x2,
                                                                                  _threshold_scAct=2,
                                                                                  target_state='A1U',
                                                                                  inc_matrix_adj=self.network.inc_matrix_adj2)
        inc_matrix_adj_UA2_act_triangle = self.get_inc_matrix_adjacency_activation(inc_matrix_col_feature=self.x2,
                                                                                  _threshold_scAct=2,
                                                                                  target_state='UA2',
                                                                                  inc_matrix_adj=self.network.inc_matrix_adj2)
        inc_matrix_adj_A1A2_act_triangle = self.get_inc_matrix_adjacency_activation(inc_matrix_col_feature=self.x2,
                                                                                  _threshold_scAct=2,
                                                                                  target_state='A1A2',
                                                                                  inc_matrix_adj=self.network.inc_matrix_adj2)

        # adj_act_edges：（节点数）表示节点i相邻激活边数量
        adj_A1U_act_edges = torch.sparse.sum(inc_matrix_adj_A1U_act_edge,dim=1).to_dense()
        adj_UA2_act_edges = torch.sparse.sum(inc_matrix_adj_UA2_act_edge,dim=1).to_dense()
        adj_A1A2_act_edges = torch.sparse.sum(inc_matrix_adj_A1A2_act_edge,dim=1).to_dense()
        adj_A1U_act_triangles = torch.sparse.sum(inc_matrix_adj_A1U_act_triangle,dim=1).to_dense()
        adj_UA2_act_triangles = torch.sparse.sum(inc_matrix_adj_UA2_act_triangle,dim=1).to_dense()
        adj_A1A2_act_triangles = torch.sparse.sum(inc_matrix_adj_A1A2_act_triangle,dim=1).to_dense()
        
        return adj_A1U_act_edges, adj_UA2_act_edges, adj_A1A2_act_edges, adj_A1U_act_triangles, adj_UA2_act_triangles, adj_A1A2_act_triangles



    def _preparing_spreading_data(self):
        old_x0 = copy.deepcopy(self.x0)
        old_x1 = copy.deepcopy(self.x1)
        true_tp = torch.zeros(self.x0.shape).to(self.DEVICE)
        return old_x0, old_x1, true_tp

    def _get_nodeid_for_each_state(self):
        UU_index = torch.where(self.x0[:, self.STATES_MAP["UU"]] == 1)[0].to(self.DEVICE, dtype=torch.long)
        A1U_index = torch.where(self.x0[:, self.STATES_MAP["A1U"]] == 1)[0].to(self.DEVICE, dtype=torch.long)
        UA2_index = torch.where(self.x0[:, self.STATES_MAP["UA2"]] == 1)[0].to(self.DEVICE, dtype=torch.long)
        A1A2_index = torch.where(self.x0[:, self.STATES_MAP["A1A2"]] == 1)[0].to(self.DEVICE, dtype=torch.long)
        return UU_index, A1U_index, UA2_index, A1A2_index


    def _dynamic_for_node(self,true_tp):
        UU_index, A1U_index, UA2_index, A1A2_index = self._get_nodeid_for_each_state()
        adj_A1U_act_edges, adj_UA2_act_edges, adj_A1A2_act_edges, adj_A1U_act_triangles, adj_UA2_act_triangles, adj_A1A2_act_triangles = self.get_adj_activate_simplex()
        # 不被一阶感染
        BETA_A1A2ToA1 = self.BETA_A1 * (1 - self.BETA_A2) / (self.BETA_A1*(1 - self.BETA_A2) + self.BETA_A2*(1 - self.BETA_A1) + 1.e-15)
        BETA_A1A2ToA2 = self.BETA_A2 * (1 - self.BETA_A1) / (self.BETA_A1*(1 - self.BETA_A2) + self.BETA_A2*(1 - self.BETA_A1) + 1.e-15)
        q_A1U = torch.pow(1 - self.BETA_A1, adj_A1U_act_edges)
        q_UA2 = torch.pow(1 - self.BETA_A2, adj_UA2_act_edges)
        q_A1A2ToA1 = torch.pow(1 - BETA_A1A2ToA1, adj_A1A2_act_edges)
        q_A1A2ToA2 = torch.pow(1 - BETA_A1A2ToA2, adj_A1A2_act_edges)
        # 不被二阶感染
        BETA_DELTA_A1A2ToA1 = self.BETA_DELTA_A1*(1 - self.BETA_DELTA_A2) / (
                    self.BETA_DELTA_A1*(1 - self.BETA_DELTA_A2) + self.BETA_DELTA_A2*(1 - self.BETA_DELTA_A1))
        BETA_DELTA_A1A2ToA2 = self.BETA_DELTA_A2*(1 - self.BETA_DELTA_A1) / (
                    self.BETA_DELTA_A1*(1 - self.BETA_DELTA_A2) + self.BETA_DELTA_A2*(1 - self.BETA_DELTA_A1))
        q_DELTA_A1U = torch.pow(1 - self.BETA_DELTA_A1, adj_A1U_act_triangles)
        q_DELTA_UA2 = torch.pow(1 - self.BETA_DELTA_A2, adj_UA2_act_triangles)
        q_DELTA_A1A2ToA1 = torch.pow(1 - BETA_DELTA_A1A2ToA1, adj_A1A2_act_triangles)
        q_DELTA_A1A2ToA2 = torch.pow(1 - BETA_DELTA_A1A2ToA2, adj_A1A2_act_triangles)
        # 一阶促进感染
        q_PRIME_A1 = torch.pow((1 - self.BETA_PRIME_A1), (adj_A1U_act_edges+adj_A1A2_act_edges))
        q_PRIME_A2 = torch.pow((1 - self.BETA_PRIME_A2), (adj_UA2_act_edges+adj_A1A2_act_edges))
        q_DELTA_PRIME_A1 = torch.pow((1 - self.BETA_DELTA_PRIME_A1), (adj_A1U_act_triangles+adj_A1A2_act_triangles))
        q_DELTA_PRIME_A2 = torch.pow((1 - self.BETA_DELTA_PRIME_A2), (adj_UA2_act_triangles+adj_A1A2_act_triangles))

        # UU不被影响的概率
        q_A1 = q_A1U * q_A1A2ToA1 * q_DELTA_A1U * q_DELTA_A1A2ToA1
        q_A2 = q_UA2 * q_A1A2ToA2 * q_DELTA_UA2 * q_DELTA_A1A2ToA2

        # 被影响的概率
        g_A1 = 1 - q_A1
        g_A2 = 1 - q_A2
        epsilon = torch.tensor(1e-7).to(g_A1)
        clamp_min = torch.tensor(0.).to(g_A1) + epsilon
        clamp_max = torch.tensor(1.).to(g_A1) - epsilon
        g_A1 = torch.clamp(g_A1, min=clamp_min, max=clamp_max)
        g_A2 = torch.clamp(g_A2, min=clamp_min, max=clamp_max)

        f_A1 = g_A1*(1-g_A2)/(g_A1*(1-g_A2)+g_A2*(1-g_A1))
        f_A2 = g_A2*(1-g_A1)/(g_A1*(1-g_A2)+g_A2*(1-g_A1))



        # 恢复，恢复迁移需要保证和为1
        recovery_prob = 1 - (1 - self.MU_A1) * (1 - self.MU_A2)
        f_MU_A1 = self.MU_A1*(1-self.MU_A2)/ (self.MU_A1*(1-self.MU_A2)+self.MU_A2*(1-self.MU_A1))
        f_MU_A2 = self.MU_A2*(1-self.MU_A1)/ (self.MU_A1*(1-self.MU_A2)+self.MU_A2*(1-self.MU_A1))

        # UU实际迁移概率
        true_tp[UU_index, self.STATES_MAP["UU"]]  = (q_A1*q_A2)[UU_index]
        true_tp[UU_index, self.STATES_MAP["A1U"]] = ((1 - q_A1 * q_A2) * f_A1)[UU_index]
        true_tp[UU_index, self.STATES_MAP["UA2"]] = ((1 - q_A1 * q_A2) * f_A2)[UU_index]
        true_tp[UU_index, self.STATES_MAP["A1A2"]] = 0
        # A1U实际迁移概率
        true_tp[A1U_index, self.STATES_MAP["UU"]]  = (self.MU_A1 * q_PRIME_A2 * q_DELTA_PRIME_A2)[A1U_index]
        true_tp[A1U_index, self.STATES_MAP["A1U"]] = ((1 - self.MU_A1) * q_PRIME_A2 * q_DELTA_PRIME_A2)[A1U_index]
        true_tp[A1U_index, self.STATES_MAP["UA2"]] = (self.MU_A1 * (1 - q_PRIME_A2 * q_DELTA_PRIME_A2))[A1U_index]
        true_tp[A1U_index, self.STATES_MAP["A1A2"]] = ((1 - self.MU_A1) * (1 - q_PRIME_A2 * q_DELTA_PRIME_A2))[A1U_index]
        # UA2实际迁移概率
        true_tp[UA2_index, self.STATES_MAP["UU"]] = (q_PRIME_A1 * q_DELTA_PRIME_A1 * self.MU_A2)[UA2_index]
        true_tp[UA2_index, self.STATES_MAP["A1U"]] = ((1 - q_PRIME_A1 * q_DELTA_PRIME_A1) * self.MU_A2)[UA2_index]
        true_tp[UA2_index, self.STATES_MAP["UA2"]] = (q_PRIME_A1 * q_DELTA_PRIME_A1 * (1 - self.MU_A2))[UA2_index]
        true_tp[UA2_index, self.STATES_MAP["A1A2"]] = ((1 - q_PRIME_A1 * q_DELTA_PRIME_A1) * (1 - self.MU_A2))[UA2_index]
        # UA2实际迁移概率
        true_tp[A1A2_index, self.STATES_MAP["UU"]] = 0
        true_tp[A1A2_index, self.STATES_MAP["A1U"]] = recovery_prob * f_MU_A1
        true_tp[A1A2_index, self.STATES_MAP["UA2"]] = recovery_prob * f_MU_A2
        true_tp[A1A2_index, self.STATES_MAP["A1A2"]] = 1 - recovery_prob

        le0_index = torch.where(true_tp.sum(dim=1) <= 0)[0]
        isnan_index = torch.where(torch.isnan(true_tp))[0]
        try:
            if len(le0_index) > 0:
                print("sum of true_tp <=0\n", le0_index)
                print("state of the node\n", self.x0[le0_index])
                raise Exception("wrong true_tp")
            if len(isnan_index) > 0:
                print("true_tp is nan\n", isnan_index)
                print("state of the node\n", self.x0[isnan_index])
                raise Exception("wrong true_tp")
        except Exception as e:
            print(e)
        pass

    def _spread(self):
        old_x0, old_x1, true_tp= self._preparing_spreading_data()
        self._dynamic_for_node(true_tp)
        new_x0 = self.get_transition_state(true_tp)
        weight = 1. * torch.ones(self.NUM_NODES).to(self.DEVICE)
        spread_result = {
            "old_x0":old_x0,
            "new_x0":new_x0,
            "true_tp":true_tp,
            "weight":weight
        }
        return spread_result

    def _run_onestep(self):
        self.BETA_A1 = self.EFF_AWARE_A1 * self.MU_A1 / self.network.AVG_K
        self.BETA_A2 = self.EFF_AWARE_A2 * self.MU_A2 / self.network.AVG_K
        self.BETA_DELTA_A1 = self.EFF_AWARE_DELTA_A1 * self.MU_A1 / self.network.AVG_K_DELTA
        self.BETA_DELTA_A2 = self.EFF_AWARE_DELTA_A2 * self.MU_A2 / self.network.AVG_K_DELTA
        self.BETA_PRIME_A1 = 1 - (1 - self.BETA_A1)**self.prime_A1
        self.BETA_PRIME_A2 = 1 - (1 - self.BETA_A2)**self.prime_A2
        self.BETA_DELTA_PRIME_A1 = 1 - (1 - self.BETA_DELTA_A1)**self.prime_A1
        self.BETA_DELTA_PRIME_A2 = 1 - (1 - self.BETA_DELTA_A2)**self.prime_A2


        spread_result = self._spread()
        return spread_result
    def get_adj_activate_simplex_dict(self):
        '''
        将adj_activate_simplex聚合dict返回
        :return:
        '''
        adj_A1U_act_edges, adj_UA2_act_edges, adj_A1A2_act_edges, adj_A1U_act_triangles, adj_UA2_act_triangles, adj_A1A2_act_triangles = self.get_adj_activate_simplex()
        adj_activate_simplex_dict = {
            "adj_A1U_act_edges": adj_A1U_act_edges.to('cpu').numpy(),
            "adj_UA2_act_edges": adj_UA2_act_edges.to('cpu').numpy(),
            "adj_A1A2_act_edges": adj_A1A2_act_edges.to('cpu').numpy(),
            "adj_A1U_act_triangles": adj_A1U_act_triangles.to('cpu').numpy(),
            "adj_UA2_act_triangles": adj_UA2_act_triangles.to('cpu').numpy(),
            "adj_A1A2_act_triangles": adj_A1A2_act_triangles.to('cpu').numpy(),
        }

        return adj_activate_simplex_dict