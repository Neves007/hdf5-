import random
import torch
from ..util.util import nodeToEdge_matrix
from mydynalearn.networks.network import Network
class ER(Network):
    def __init__(self, net_config):
        super().__init__(net_config)
        pass

    # 边界矩阵B
    def _create_edges(self):
        edges = set()
        while len(edges) < self.NUM_EDGES:
            edge = random.sample(range(self.NUM_NODES), 2)
            edge.sort()
            edges.add(tuple(edge))
        edges = torch.asarray(list(edges))  # 最终的边
        self.__setattr__("edges", edges)


    def _update_adj(self):
        # inc_matrix_0：节点和节点的关联矩阵
        # networkx的边先对边进行预处理，无相边会有问题。
        inverse_matrix = torch.asarray([[0, 1], [1, 0]])
        edges_inverse = torch.mm(self.edges, inverse_matrix)  # 对调两行
        inc_matrix_adj0 = torch.sparse_coo_tensor(indices=torch.cat([self.edges.T, edges_inverse.T], dim=1),
                                                 values=torch.ones(2 * self.NUM_EDGES),
                                                 size=(self.NUM_NODES, self.NUM_NODES))
        # inc_matrix_1：节点和边的关联矩阵
        inc_matrix_adj1 = nodeToEdge_matrix(self.nodes, self.edges)
        inc_matrix_adj1 = inc_matrix_adj1.to_sparse()

        # 随机断边
        self.__setattr__("inc_matrix_adj0",inc_matrix_adj0)
        self.__setattr__("inc_matrix_adj1",inc_matrix_adj1)


    def to_device(self, device):
        self.DEVICE = device
        self.nodes = self.nodes.to(self.DEVICE)
        self.edges = self.edges.to(self.DEVICE)
        self.NUM_NODES = self.NUM_NODES
        self.NUM_EDGES = self.NUM_EDGES
        self.AVG_K = self.AVG_K

        self.inc_matrix_adj0 = self.inc_matrix_adj0.to(self.DEVICE)
        self.inc_matrix_adj1 = self.inc_matrix_adj1.to(self.DEVICE)

    def _unpack_inc_matrix_adj_info(self):
        return self.inc_matrix_adj0, self.inc_matrix_adj1

    def _update_topology_info(self):
        self.AVG_K = 2 * len(self.edges) / self.NUM_NODES

    def build_dataset(self):
        nodes = torch.arange(self.NUM_NODES)
        self.__setattr__("nodes", nodes)
        NUM_EDGES = int(self.AVG_K * self.NUM_NODES / 2)
        self.__setattr__("NUM_EDGES", NUM_EDGES)
        self._create_edges()
        self._update_topology_info()
        self._update_adj()
        dataset = {
            "nodes": self.nodes,
            "edges": self.edges,
            "NUM_NODES": self.NUM_NODES,
            "AVG_K": self.AVG_K,
            "inc_matrix_adj0": self.inc_matrix_adj0,
            "inc_matrix_adj1": self.inc_matrix_adj1,
        }
        self.set_dataset(dataset)
