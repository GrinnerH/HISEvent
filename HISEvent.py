import networkx as nx
from SE import SE
from itertools import combinations, chain
import numpy as np

# 算法1
def search_stable_points(embeddings, max_num_neighbors = 200):
    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    corr_matrix_sorted_indices = np.argsort(corr_matrix)
    
    all_1dSEs = []
    seg = None
    for i in range(max_num_neighbors):
        dst_ids = corr_matrix_sorted_indices[:, -(i+1)]
        knn_edges = [(s+1, d+1, corr_matrix[s, d]) \
            for s, d in enumerate(dst_ids) if corr_matrix[s, d] > 0] # (s+1, d+1): +1 as node indexing starts from 1 instead of 0
        if i == 0:
            g = nx.Graph()
            g.add_weighted_edges_from(knn_edges)
            seg = SE(g)
            all_1dSEs.append(seg.calc_1dSE())
        else:
            all_1dSEs.append(seg.update_1dSE(all_1dSEs[-1], knn_edges))
    
    #print('all_1dSEs: ', all_1dSEs)
    stable_indices = []
    for i in range(1, len(all_1dSEs) - 1):
        if all_1dSEs[i] < all_1dSEs[i - 1] and all_1dSEs[i] < all_1dSEs[i + 1]:
            stable_indices.append(i)
    if len(stable_indices) == 0:
        print('No stable points found after checking k = 1 to ', max_num_neighbors)
        return 0, 0
    else:
        stable_SEs = [all_1dSEs[index] for index in stable_indices]
        index = stable_indices[stable_SEs.index(min(stable_SEs))]
        print('stable_indices: ', stable_indices)
        print('stable_SEs: ', stable_SEs)
        print('First stable point: k = ', stable_indices[0]+1, ', correspoding 1dSE: ', stable_SEs[0]) # n_neighbors should be index + 1
        print('Global stable point within the searching range: k = ', index + 1, \
            ', correspoding 1dSE: ', all_1dSEs[index]) # n_neighbors should be index + 1
    return stable_indices[0]+1, index + 1 # first stable point, global stable point

def get_graph_edges(attributes):
    attr_nodes_dict = {}
    for i, l in enumerate(attributes):
        for attr in l:
            if attr not in attr_nodes_dict:
                attr_nodes_dict[attr] = [i+1] # node indexing starts from 1
            else:
                attr_nodes_dict[attr].append(i+1)

    for attr in attr_nodes_dict.keys():
        attr_nodes_dict[attr].sort()

    graph_edges = []
    for l in attr_nodes_dict.values():
        graph_edges += list(combinations(l, 2))
    return list(set(graph_edges))

def get_knn_edges(embeddings, default_num_neighbors):
    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    corr_matrix_sorted_indices = np.argsort(corr_matrix)
    knn_edges = []
    for i in range(default_num_neighbors):
        dst_ids = corr_matrix_sorted_indices[:, -(i+1)]
        knn_edges += [(s+1, d+1) if s < d else (d+1, s+1) \
            for s, d in enumerate(dst_ids) if corr_matrix[s, d] > 0] # (s+1, d+1): +1 as node indexing starts from 1 instead of 0
    return list(set(knn_edges))

def get_global_edges(attributes, embeddings, default_num_neighbors, e_a = True, e_s = True):
    graph_edges, knn_edges = [], []
    if e_a == True:
        graph_edges = get_graph_edges(attributes)
    if e_s == True:
        knn_edges = get_knn_edges(embeddings, default_num_neighbors)
    return list(set(knn_edges + graph_edges))

def get_subgraphs_edges(clusters, graph_splits, weighted_global_edges):
    '''
    get the edges of each subgraph

    clusters: a list containing the current clusters, each cluster is a list of nodes of the original graph
    graph_splits: a list of (start_index, end_index) pairs, each (start_index, end_index) pair indicates a subset of clusters, 
        which will serve as the nodes of a new subgraph
    weighted_global_edges: a list of (start node, end node, edge weight) tuples, each tuple is an edge in the original graph

    return: all_subgraphs_edges: a list containing the edges of all subgraphs
    '''
    all_subgraphs_edges = []
    # 遍历 graph_splits 列表中的每个 (start_index, end_index) 元组
    for split in graph_splits:
        subgraph_clusters = clusters[split[0]:split[1]]
        subgraph_nodes = list(chain(*subgraph_clusters))
        # 从全局加权边列表中筛选出两个端点都在子图节点列表中的边
        '''例如：
        weighted_global_edges = [(1, 2, 0.5), (2, 3, 0.6), (3, 4, 0.7), (1, 5, 0.8)]
        subgraph_nodes = [1, 2, 3]
        只有 (1, 2, 0.5) 和 (2, 3, 0.6) 这两条边的两个端点都在子图节点列表中
        '''
        subgraph_edges = [edge for edge in weighted_global_edges if edge[0] in subgraph_nodes and edge[1] in subgraph_nodes]
        all_subgraphs_edges.append(subgraph_edges)
    return all_subgraphs_edges

# 算法2
def hier_2D_SE_mini(weighted_global_edges, n_messages, n = 100):
    '''
    hierarchical 2D SE minimization
    '''
    # weighted_global_edges：全局加权边列表，包含消息图的所有边及其权重
    # n_messages：消息的数量，即消息图的节点数量
    # n：超参数，用于控制每次处理的聚类数量，默认为100

    #初始化迭代次数
    ite = 1
    # initially, each node (message) is in its own cluster
    # node encoding starts from 1
    # 初始化聚类，将每个消息（节点）初始化为一个独立的聚类。
    # 每个聚类用一个列表表示，列表中只有一个元素，即该消息的编号（从1开始）
    clusters = [[i+1] for i in range(n_messages)]
    while True:
        print('\n=========Iteration ', str(ite), '=========')
        # 获取当前聚类的数量
        n_clusters = len(clusters)

        # 将当前的聚类集合划分为多个小子集，每个子集的大小是min(s+n, n_clusters)，最多为n。
        # graph_splits是一个列表，每个元素是一个元组[s, e)，表示一个子集的起始索引和结束索引（不包含e）
        graph_splits = [(s, min(s+n, n_clusters)) for s in range(0, n_clusters, n)] # [s, e)
        
        # 根据划分的子集和全局加权边列表，获取每个子图的边列表。
        # 这时应该存一些边的两个端点都不在同一个子图中。
        all_subgraphs_edges = get_subgraphs_edges(clusters, graph_splits, weighted_global_edges)
        # 保存当前的聚类结果，用于后续检查聚类是否收敛
        last_clusters = clusters
        clusters = []
        # 遍历每个子图的边列表，把边列表构建成子图，计算最小2D SE
        for i, subgraph_edges in enumerate(all_subgraphs_edges):
            print('\tSubgraph ', str(i+1))
            # 创建一个无向图对象g，并添加子图的边。这里使用了NetworkX库的Graph类
            g = nx.Graph()
            g.add_weighted_edges_from(subgraph_edges)

            # 结构熵最小化
            seg = SE(g)
            # 初始化seg对象的division属性，将当前子图对应的聚类划分存储在其中。
            # division是一个字典，键是聚类的编号，值是聚类中的节点列表
            seg.division = {j: cluster for j, cluster in enumerate(last_clusters[graph_splits[i][0]:graph_splits[i][1]])}
            # 调用seg对象的add_isolates方法，将任何孤立节点添加到图中。
            # 该方法的具体实现如下：
            # 首先获取division中所有节点的列表，并排序；然后获取图中（已经通过边连接）所有节点的列表，也排序；如果两个列表不相等，说明存在孤立节点，将这些孤立节点添加到图中
            seg.add_isolates()

            # 遍历division中的每个聚类，将聚类中的每个节点的'comm'属性设置为该聚类的编号
            for k in seg.division.keys():
                for node in seg.division[k]:
                    seg.graph.nodes[node]['comm'] = k
            
            # 计算每个社区的体积、割、社区节点SE和叶节点SE，并存储在struc_data属性中。
            # 首先清空struc_data字典；然后遍历division中的每个社区，计算其体积、割等信息；最后将这些信息存储在struc_data字典中
            seg.update_struc_data()

            # 调用seg对象的update_struc_data_2d方法，计算每对社区合并后的体积、割、社区节点SE和叶节点SE，并存储在struc_data_2d属性中。
            # 首先清空struc_data_2d字典；然后遍历所有可能的社区对，计算合并后的相关信息；最后将这些信息存储在struc_data_2d字典中
            seg.update_struc_data_2d()

            # 调用seg对象的update_division_MinSE方法，贪心更新编码树以最小化2D SE。
            # 定义一个内部函数Mg_operator，用于计算合并两个社区引起的SE变化；
            # 然后进入一个循环，不断寻找能使SE最大程度降低的两个社区进行合并，直到SE无法再降低；
            # 在合并过程中，更新division、struc_data和struc_data_2d属性
            seg.update_division_MinSE()

            # 将seg对象的division属性中的所有聚类添加到clusters列表中
            clusters += list(seg.division.values())
        # 如果graph_splits列表的长度为1，说明所有的聚类都在一个子集中，此时跳出循环
        if len(graph_splits) == 1:
            break
        # 如果当前的聚类结果与上一次迭代结束时的聚类结果相同，说明聚类已经收敛，无法再进行合并。
        # 此时将n增大为原来的2倍，以便在后续迭代中考虑更多的聚类
        if clusters == last_clusters:
            n *= 2
    return clusters
