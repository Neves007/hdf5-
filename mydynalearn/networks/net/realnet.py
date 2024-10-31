import random
import numpy as np
import torch
from scipy.special import comb
from ..util.util import nodeToEdge_matrix,nodeToTriangle_matrix
from mydynalearn.networks.network import Network
from mydynalearn.networks.util.real_network import generate_real_network
import os
import pickle
class Realnet(Network):
    def __init__(self, config):
        super().__init__(config)

    def _update_adj(self):
        # inc_matrix_0：节点和节点的关联矩阵
        # 先对边进行预处理，无相边会有问题。
        inverse_matrix=torch.tensor([[0, 1], [1, 0]],dtype=torch.long)
        edges_inverse = torch.mm(self.edges,inverse_matrix) # 对调两行
        # inc_matrix_0：节点和节点的关联矩阵，即邻接矩阵
        inc_matrix_adj0 = torch.sparse_coo_tensor(indices=torch.cat([self.edges.T,edges_inverse.T],dim=1),
                                              values=torch.ones(2*self.NUM_EDGES),
                                              size=(self.NUM_NODES,self.NUM_NODES))
        # inc_matrix_1：节点和边的关联矩阵
        inc_matrix_adj1 = nodeToEdge_matrix(self.nodes, self.edges)
        inc_matrix_adj1 = inc_matrix_adj1.to_sparse()

        # inc_matrix_2：节点和高阶边的关联矩阵
        inc_matrix_adj2 = nodeToTriangle_matrix(self.nodes, self.triangles)
        inc_matrix_adj2 = inc_matrix_adj2.to_sparse()
        # 随机断边
        inc_matrix_adj_info = {
            "inc_matrix_adj0":inc_matrix_adj0,
            "inc_matrix_adj1":inc_matrix_adj1,
            "inc_matrix_adj2":inc_matrix_adj2
        }
        self.set_attr(inc_matrix_adj_info)
        self.__setattr__("inc_matrix_adj_info",inc_matrix_adj_info)


    def to_device(self,device):
        self.DEVICE = device
        self.nodes = self.nodes.to(device)
        self.edges = self.edges.to(device)
        self.triangles = self.triangles.to(device)
        self.NUM_NODES = self.NUM_NODES
        self.NUM_EDGES = self.NUM_EDGES
        self.NUM_TRIANGLES = self.NUM_TRIANGLES
        self.AVG_K = self.AVG_K
        self.AVG_K_DELTA = self.AVG_K_DELTA

        self.inc_matrix_adj0 = self.inc_matrix_adj0.to(device)
        self.inc_matrix_adj1 = self.inc_matrix_adj1.to(device)
        self.inc_matrix_adj2 = self.inc_matrix_adj2.to(device)
    def _unpack_inc_matrix_adj_info(self):
        return self.inc_matrix_adj0, self.inc_matrix_adj1, self.inc_matrix_adj2

    def create_datawork(self):

        netsourve_file = os.path.join(self.REALNET_DATA_PATH, self.REALNET_SOURCEDATA_FILENAME)
        nodes, edges, triangles = generate_real_network(netsourve_file)
        NUM_NODES = nodes.shape[0]
        NUM_EDGES = edges.shape[0]
        NUM_TRIANGLES = triangles.shape[0]
        AVG_K = 2 * len(edges) / NUM_NODES
        AVG_K_DELTA = 3 * len(triangles) / NUM_NODES
        net_info = {"nodes": nodes,
                    "edges": edges,
                    "triangles": triangles,
                    "NUM_NODES": NUM_NODES,
                    "NUM_EDGES": NUM_EDGES,
                    "NUM_TRIANGLES": NUM_TRIANGLES,
                    "AVG_K": AVG_K,
                    "AVG_K_DELTA": AVG_K_DELTA,
                    }
        self.__setattr__("net_info",net_info)
        self.set_attr(net_info)

    def build(self):
        self.create_datawork()
        self._update_adj()