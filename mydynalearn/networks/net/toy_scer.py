import random
import numpy as np
import torch
from scipy.special import comb
from ..util.util import nodeToEdge_matrix,nodeToTriangle_matrix
from mydynalearn.networks.network import Network
class ToySCER():
    def __init__(self, net_config):
        self.net_config = net_config
        self.DEVICE = net_config.DEVICE
        self.MAX_DIMENSION = self.net_config.MAX_DIMENSION
        self.NUM_NODES = self.net_config.NUM_NODES

        self.net_info = self.get_net_info()  # 网络信息
        self._set_net_info()
        self.inc_matrix_adj_info = self._get_adj()  # 关联矩阵
        self.set_inc_matrix_adj_info()
        self.to_device()
        pass
    def set_inc_matrix_adj_info(self):
        self.inc_matrix_adj0 = self.inc_matrix_adj_info["inc_matrix_adj0"]
        self.inc_matrix_adj1 = self.inc_matrix_adj_info["inc_matrix_adj1"]
        self.inc_matrix_adj2 = self.inc_matrix_adj_info["inc_matrix_adj2"]

    def _set_net_info(self):
        self.nodes = self.net_info["nodes"]
        self.edges = self.net_info["edges"]
        self.triangles = self.net_info["triangles"]
        self.NUM_NODES = self.net_info["NUM_NODES"]
        self.NUM_EDGES = self.net_info["NUM_EDGES"]
        self.NUM_TRIANGLES = self.net_info["NUM_TRIANGLES"]
        self.AVG_K = self.net_info["AVG_K"]
        self.AVG_K_DELTA = self.net_info["AVG_K_DELTA"]

    def get_net_info(self):

        nodes =  torch.arange(self.NUM_NODES)
        edges = torch.tensor([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                                       (1, 2), (1, 7),
                                       (2, 7), (4, 5), (5, 6)])
        triangles = torch.tensor([(0, 4, 5), (0, 5, 6), (1, 2, 7)])
        NUM_NODES = self.NUM_NODES
        NUM_EDGES = edges.shape[0]
        NUM_TRIANGLES = triangles.shape[0]
        AVG_K = 2*len(edges)/NUM_NODES
        AVG_K_DELTA = 3*len(triangles)/NUM_NODES
        net_info = {"nodes": nodes,
                    "edges": edges,
                    "triangles": triangles,
                    "NUM_NODES": NUM_NODES,
                    "NUM_EDGES": NUM_EDGES,
                    "NUM_TRIANGLES": NUM_TRIANGLES,
                    "AVG_K": AVG_K,
                    "AVG_K_DELTA": AVG_K_DELTA
                    }
        return net_info
    def _get_adj(self):
        # inc_matrix_0：节点和节点的关联矩阵
        # 先对边进行预处理，无相边会有问题。
        inverse_matrix=torch.asarray([[0, 1], [1, 0]])
        edges_inverse = torch.mm(self.edges,inverse_matrix) # 对调两行
        inc_matrix_adj0 = torch.sparse_coo_tensor(indices=torch.cat([self.edges.T,edges_inverse.T],dim=1),
                                              values=torch.ones(2*self.NUM_EDGES),
                                              size=(self.NUM_NODES,self.NUM_NODES))
        # inc_matrix_1：节点和边的关联矩阵
        inc_matrix_adj1 = nodeToEdge_matrix(self.nodes, self.edges)
        inc_matrix_adj1 = inc_matrix_adj1.to_sparse()

        inc_matrix_adj2 = nodeToTriangle_matrix(self.nodes, self.triangles)
        inc_matrix_adj2 = inc_matrix_adj2.to_sparse()
        # 随机断边
        inc_matrix_adj_info = {
            "inc_matrix_adj0":inc_matrix_adj0,
            "inc_matrix_adj1":inc_matrix_adj1,
            "inc_matrix_adj2":inc_matrix_adj2
        }
        return inc_matrix_adj_info
    def to_device(self,device):
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