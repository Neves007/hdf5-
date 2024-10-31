import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations,permutations
import torch
import matplotlib.pyplot as plt
from networkx.algorithms.clique import enumerate_all_cliques

def trans_node_index(data_df):
    nodes = np.sort(pd.concat([data_df['i'],data_df['j']], axis=0, ignore_index=True).unique())
    nodes_id_dict= {}
    for index, name in enumerate(nodes):
        nodes_id_dict[name] = index
    data_df['i'] = [nodes_id_dict[x] for x in data_df['i']]
    data_df['j'] = [nodes_id_dict[x] for x in data_df['j']]

    return data_df

def get_unique_triangles(triangles):
    unique_triangles = set()
    for tri in triangles:
        tri = list(tri)
        tri.sort()
        unique_triangles.add(tuple(tri))
    unique_triangles = [list(tri) for tri in unique_triangles]
    unique_triangles = torch.tensor(np.asarray(unique_triangles), dtype=torch.long)
    return unique_triangles
def generate_real_network(read_file):
    ######读取文件，聚合成每行为: t [(i1，j1),(i2,j2)```]的形式
    ## 第一列时间戳，第二列节点i，第三列节点j交互的个体。
    data_df = pd.read_csv(read_file,delimiter=' ',names=['t','i','j','m','n']).fillna(0)#读取一个csv格式的配置文件，并转化为数据框形式。
    data_df=data_df[["t","i","j"]]
    # 转换节点序号
    data_df = trans_node_index(data_df)
    data_s=data_df.groupby('t',sort=False).apply(lambda x:list(zip(x.i,x.j)))
    t=data_s.index.tolist()
    data_df=pd.DataFrame(t,columns=['t'])
    data_df['ij']=data_s.values
    #display(data_df)
    length=len(data_df)
    max_t=data_df.iloc[len(data_df)-1].t

    ######把每五分钟的数据聚合成一组
    data_combi_final=[]
    for tt in data_df.t:
        if tt+300<=max_t:
            data_combi=[]
            df_help=data_df[data_df.t>=tt]
            for index,data in df_help.iterrows():
                if data.t<=tt+300:
                    data_combi=data_combi+data.ij
                else:
                    break
            data_combi_final.append(data_combi)
        else:
            break
    #display(data_combi_final)

    #####画图，找出所有2-simplex和1-simplex
    G = nx.Graph()
    G.add_edges_from(data_combi_final[0])
    all_cliques= nx.enumerate_all_cliques(G)
    two_cliques=[x for x in all_cliques if len(x)==2]
    all_cliques= nx.enumerate_all_cliques(G)
    triad_cliques=[x for x in all_cliques if len(x)==3]
    #生成初始数据框
    two_cliques_df=pd.DataFrame(two_cliques,columns=['i','j'])
    triad_cliques_df=pd.DataFrame(triad_cliques,columns=['i','j','l'])
    #开始遍历更新数据框
    for g in  data_combi_final:
        G = nx.Graph()
        G.add_edges_from(g)
        new_all_cliques= nx.enumerate_all_cliques(G)
        new_two_cliques=[x for x in new_all_cliques if len(x)==2]
        new_all_cliques= nx.enumerate_all_cliques(G)
        new_triad_cliques=[x for x in new_all_cliques if len(x)==3]
        new_two_cliques_df=pd.DataFrame(new_two_cliques,columns=['i','j'])
        new_triad_cliques_df=pd.DataFrame(new_triad_cliques,columns=['i','j','l'])
        two_cliques_df = pd.concat([two_cliques_df, new_two_cliques_df], ignore_index=True)
        triad_cliques_df = pd.concat([triad_cliques_df, new_triad_cliques_df], ignore_index=True)
    #display(two_cliques_df)
    #display(triad_cliques_df)

    #####选出前20%的simplex,并分别得到所有二元组和三元组
    two_cliques_df_result= two_cliques_df.groupby(['i', 'j'])["i"].count().reset_index(name="count")
    triad_cliques_df_result=triad_cliques_df.groupby(['i', 'j','l'])["i"].count().reset_index(name="count")
    #display(two_cliques_df_result)
    #display(triad_cliques_df_result)
    result=pd.concat([two_cliques_df_result,triad_cliques_df_result],sort=False)
    result=result.sort_values("count",ascending=False).reset_index(drop=True)
    result=result.head(int(len(result)*0.2))
    result1=result[np.isnan(result['l'])]
    result2=result[~np.isnan(result['l'])].astype(np.int64)
    result1=result1[['i','j']]
    result2=result2[['i','j','l']]#.astype(str)
    #display(result1)
    #display(result2)
    result1['ij']=result1.apply(tuple,axis=1)
    result2['ijl']=result2.apply(tuple,axis=1)

    ####组成最终网络 并得到三元组和邻居列表
    link=result1.ij.tolist()
    tri=result2.ijl.tolist()
    #处理一下三元组
    tri_link=[]
    for tr in tri:
        for i,j in combinations(tr,2):
            tri_link.append((i,j))
    #画图
    G = nx.Graph()
    G.add_edges_from(link)
    G.add_edges_from(tri_link)
    #nx.draw(G)
    #plt.show()
    #获取最大连通图
    if not nx.is_connected(G):
        giant = nx.subgraph(G,max(nx.connected_components(G), key=len),)
        G = nx.Graph(giant)
    #获取邻居集合,三元组
    node_neighbors_dict = {}
    for n in G.nodes():
        node_neighbors_dict[n] = G[n].keys()
    triangles_list=tri
    N=G.order()
    triangles = get_unique_triangles(triangles_list)
    nodes = torch.tensor(np.asarray(G.nodes), dtype=torch.long)
    edges = torch.tensor(np.asarray(G.edges), dtype=torch.long)
    nodes_id = torch.arange(nodes.shape[0], dtype=torch.long)
    edges = torch.where(edges.view(-1,1)==nodes.view(1,-1))[1].view(edges.shape)
    triangles = torch.where(triangles.view(-1,1)==nodes.view(1,-1))[1].view(triangles.shape)



    return nodes_id, edges, triangles