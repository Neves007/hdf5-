import torch
from ..util.util import nodeToEdge_matrix,nodeToTriangle_matrix
from mydynalearn.networks.network import Network
import networkx as nx
# todo：相邻关系
    # inc_matrix_adj_info
# todo: 管理过程
    # 保存和读取
    # 输出

class SF(Network):
    def __init__(self,net_config):
        super().__init__(net_config)
        self.set_attr(self.net_config)
        pass

    def set_inc_matrix_adj_info(self):
        self.inc_matrix_adj0 = self.inc_matrix_adj_info["inc_matrix_adj0"]
        self.inc_matrix_adj1 = self.inc_matrix_adj_info["inc_matrix_adj1"]
        self.inc_matrix_adj2 = self.inc_matrix_adj_info["inc_matrix_adj2"]



    def compute_pcum_from_plink(self, pnode):
        pcum = torch.cat([torch.tensor([0], device=self.DEVICE), torch.cumsum(pnode[:, 1] / torch.sum(pnode[:, 1]), dim=0)])
        return pcum

    def _init_network(self):
        # Create initial adjacency matrix on cuda
        A0 = torch.ones((self.N0, self.N0), device=self.DEVICE)
        A0 = A0 - torch.diag(torch.diag(A0))
        A = torch.zeros((self.NUM_NODES, self.NUM_NODES), device=self.DEVICE)
        A[:self.N0, :self.N0] = A0

        # pnode：Dij的稀疏表示
        # Extract upper triangular part of Dij and find non-zero elements
        D_list = torch.sum(A,dim=1)
        non_zero_indices = torch.nonzero(D_list > 0)
        pnode = torch.cat([non_zero_indices, D_list[non_zero_indices]], dim=1)

        # 由plink的度分布决定的累积概率
        pcum = self.compute_pcum_from_plink(pnode)


        nodes = torch.arange(self.NUM_NODES)
        edges = set()
        # 根据平均度计算边和三角形的数量
        attr = {
            "A": A,
            "pnode":pnode,
            "pcum":pcum,
            "nodes": nodes,
            "edges": edges,
        }
        self.set_attr(attr)

    def _add_a_node(self, new_node_i):
        self.A[new_node_i, new_node_i] = 0

        # Generate ntri random numbers
        dummy = torch.rand(self.nlink, device=self.DEVICE)

        # 根据累积概率向量 (pcum) 和随机生成的值 (dummy)，确定哪些边被选中。
        diffs = self.pcum.view(-1, 1) - dummy.view(1, -1)  # 计算累积概率 pcum 和随机生成的值 dummy 之间的差异。
        temp = torch.diff(torch.sign(diffs), dim=0)  # 沿着维度 0（行）计算相邻元素之间的差异。
        attached_link_index_list = torch.nonzero(temp != 0)[:, 0]  # 确定符号变化的行索引（即找到随机值在哪个累积概率区间中）。

        # 确保选择的ntri个连边里没有重复节点
        if len(torch.unique(self.pnode[attached_link_index_list, :2])) == 2 * self.ntri:
            isw = 1
        return attached_link_index_list

    def _update_adj_info(self, new_node_i, inode, jnode):
        # Update A
        self.A[new_node_i, inode] = 1
        self.A[new_node_i, jnode] = 1
        self.A[inode, new_node_i] = 1
        self.A[jnode, new_node_i] = 1

        # Update Dij
        self.Dij[new_node_i, inode] = 1
        self.Dij[new_node_i, jnode] = 1
        self.Dij[inode, new_node_i] = 1
        self.Dij[jnode, new_node_i] = 1
        self.Dij[inode, jnode] += 1
        self.Dij[jnode, inode] += 1


    def _update_edges_and_triangles(self, new_node_i, inode, jnode):
        """
        更新 edges 和 triangles 集合
        :param new_node_i: 新添加的节点的索引
        :param idx: 连接到新节点的边的索引
        """
        self.edges.add(tuple(sorted([inode, jnode])))
        self.triangles.add(tuple(sorted([new_node_i, inode, jnode])))

    def _update_topology_info(self):
        # 根据建立的拓扑结构更新网络信息
        # 步骤 1：获取节点
        nodes = list(self.G.nodes)  # 节点列表
        edges = list(self.G.edges)
        nodes_tensor = torch.tensor(nodes, dtype=torch.long)  # 转为torch tensor
        edges_tensor = torch.tensor(edges, dtype=torch.long)  # 转为torch tensor

        NUM_EDGES = edges_tensor.shape[0]
        AVG_K = 2 * len(edges_tensor) / self.NUM_NODES

        net_info = {"nodes": nodes_tensor,
                    "edges": edges_tensor,
                    "NUM_EDGES": NUM_EDGES,
                    "AVG_K": AVG_K}
        self.__setattr__("net_info",net_info)
        self.set_attr(net_info)

    def _add_new_nodes(self):
        # 添加新节点
        for new_node_i in range(self.N0, self.NUM_NODES):
            attached_link_index_list = self._add_a_node(new_node_i)
            for link_index in range(len(attached_link_index_list)):
                # 新增的邻居节点
                inode = int(self.pnode[attached_link_index_list[link_index], 0].item())
                jnode = int(self.pnode[attached_link_index_list[link_index], 1].item())
                # 更新拓扑结构
                self._update_edges_and_triangles(new_node_i, inode, jnode)
                self._update_adj_info(new_node_i, inode, jnode)

                # 更新参数
                self.pnode = torch.cat(
                    [self.pnode, torch.tensor([[new_node_i, inode, 1], [new_node_i, jnode, 1]], device=self.DEVICE)],
                    dim=0)
                self.pnode[attached_link_index_list[link_index], 2] = self.Dij[inode, jnode].item()
                self.pcum = self.compute_pcum_from_plink(self.pnode)



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
        inc_matrix_adj_info = {
            "inc_matrix_adj0":inc_matrix_adj0,
            "inc_matrix_adj1":inc_matrix_adj1
        }
        # 随机断边
        self.set_attr(inc_matrix_adj_info)
        self.__setattr__("inc_matrix_adj_info",inc_matrix_adj_info)

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

    def build(self):
        # 生成无标度网络
        G = nx.barabasi_albert_graph(self.NUM_NODES, self.nlink)
        self.__setattr__("G", G)
        self._update_topology_info()
        self._update_adj()

