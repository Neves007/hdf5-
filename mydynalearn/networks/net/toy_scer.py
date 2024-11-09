import torch
from ..util.util import nodeToEdge_matrix,nodeToTriangle_matrix
from mydynalearn.networks.network import Network
class ToySCER(Network):
    def __init__(self, net_config):
        super().__init__(net_config)

    def _update_adj(self):
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
        self.inc_matrix_adj0 = inc_matrix_adj0
        self.inc_matrix_adj1 = inc_matrix_adj1
        self.inc_matrix_adj2 = inc_matrix_adj2
    def unpack_inc_matrix_adj_info(self):
        if not hasattr(self, "inc_matrix_adj0"):
            self.load() 
        return self.inc_matrix_adj0, self.inc_matrix_adj1, self.inc_matrix_adj2
    def build_dataset(self):
        self.NUM_NODES = 7
        self.nodes = torch.arange(self.NUM_NODES, device=self.DEVICE)
        self.edges = torch.tensor([(0, 1), (0, 2), (0, 4), (0, 5), (0, 6),
                              (1, 2),(2, 3), (3, 4), (4, 5)])
        self.triangles = torch.tensor([(0, 1, 2), (0, 4, 5)])
        self.NUM_EDGES = self.edges.shape[0]
        self.NUM_TRIANGLES = self.triangles.shape[0]
        self.AVG_K = 2 * self.NUM_EDGES / self.NUM_NODES
        self.AVG_K_DELTA = 3 * self.NUM_TRIANGLES / self.NUM_NODES
        self._update_adj()
        dataset = {
            "nodes": self.nodes,
            "edges": self.edges,
            "triangles": self.triangles,
            "NUM_NODES": self.NUM_NODES,
            "NUM_EDGES": self.NUM_EDGES,
            "NUM_TRIANGLES": self.NUM_TRIANGLES,
            "NUM_NEIGHBOR_NODES": self.inc_matrix_adj0.sum(dim=1).to_dense(),
            "NUM_NEIGHBOR_EDGES": self.inc_matrix_adj1.sum(dim=1).to_dense(),
            "NUM_NEIGHBOR_TRIANGLES": self.inc_matrix_adj2.sum(dim=1).to_dense(),
            "AVG_K": self.AVG_K,
            "AVG_K_DELTA": self.AVG_K_DELTA,
            "inc_matrix_adj0": self.inc_matrix_adj0,
            "inc_matrix_adj1": self.inc_matrix_adj1,
            "inc_matrix_adj2": self.inc_matrix_adj2,
        }
        self.set_dataset(dataset)