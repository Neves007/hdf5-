import torch
def nodeToEdge_matrix(nodes, edges, one_indexed=False):
    sigma = torch.zeros((len(nodes), len(edges)), dtype=torch.float)
    offset = int(one_indexed)  # 索引从0开始
    # 边索引j
    j = 0
    # oriented
    for edge in edges:
        x, y = edge
        sigma[x - offset][j] = 1
        sigma[y - offset][j] = 1
        j += 1
    return sigma

def nodeToTriangle_matrix(nodes, triangles, one_indexed=False):
    sigma = torch.zeros((len(nodes), len(triangles)), dtype=torch.float)
    offset = int(one_indexed)  # 索引从0开始
    # 边索引j
    j = 0
    # oriented
    for triangle in triangles:
        x, y, z = triangle
        sigma[x - offset][j] = 1
        sigma[y - offset][j] = 1
        sigma[z - offset][j] = 1
        j += 1
    return sigma