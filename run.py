from HISEvent import hier_2D_SE_mini, get_global_edges, search_stable_points
from utils import evaluate, decode
from datetime import datetime
import math
import numpy as np
import pickle
import pandas as pd
import os
from os.path import exists

def get_stable_point(path):
    stable_point_path = path + 'stable_point.pkl'
    if not exists(stable_point_path):
        embeddings_path = path + 'SBERT_embeddings.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        first_stable_point, global_stable_point = search_stable_points(embeddings)
        stable_points = {'first': first_stable_point, 'global': global_stable_point}
        with open(stable_point_path, 'wb') as fp:
            pickle.dump(stable_points, fp)
        print('stable points stored.')

    with open(stable_point_path, 'rb') as f:
        stable_points = pickle.load(f)
    print('stable points loaded.')
    return stable_points

def run_hier_2D_SE_mini_Event2012_open_set(n = 400, e_a = True, e_s = True, test_with_one_block = True):
    save_path = './data/Event2012/open_set/'
  
    if test_with_one_block:
        blocks = [20]
    else:
        blocks = [i+1 for i in range(21)]
    for block in blocks:
        print('\n\n====================================================')
        print('block: ', block)
        print(datetime.now().strftime("%H:%M:%S"))

        folder = f'{save_path}{block}/'
        
        # load message embeddings
        embeddings_path = folder + 'SBERT_embeddings.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        df_np = np.load(f'{folder}{block}.npy', allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=["original_index", "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
                "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", 
                "words", "filtered_words", "sampled_words", "date"])
        all_node_features = [[str(u)] + \
            [str(each) for each in um] + \
            [h.lower() for h in hs] + \
            e \
            for u, um, hs, e in \
            zip(df['user_id'], df['user_mentions'],  df['hashtags'], df['entities'])]
        
        print('all_node_features: ', all_node_features)
        
        #找到该block下的局部稳定点和全局稳定点
        stable_points = get_stable_point(folder)
        print('stable_points: ', stable_points)
        if e_a == False: # only rely on e_s (semantic-similarity-based edges)
            default_num_neighbors = stable_points['global']
        else:
            default_num_neighbors = stable_points['first']
        if default_num_neighbors == 0: 
            default_num_neighbors = math.ceil((len(embeddings)/1000)*10)
        
        global_edges = get_global_edges(all_node_features, embeddings, default_num_neighbors, e_a = e_a, e_s = e_s)
        # print('global_edges: ', global_edges)

        corr_matrix = np.corrcoef(embeddings)
        np.fill_diagonal(corr_matrix, 0)
        print('corr_matrix: ', corr_matrix)
        weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
            if corr_matrix[edge[0]-1, edge[1]-1] > 0] # node encoding starts from 1
        
        # print('weighted_global_edges: ', weighted_global_edges)
        
        division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n = n)
        print(datetime.now().strftime("%H:%M:%S"))

        prediction = decode(division)

        labels_true = df['event_id'].tolist()
        n_clusters = len(list(set(labels_true)))
        print('n_clusters gt: ', n_clusters)

        nmi, ami, ari = evaluate(labels_true, prediction)
        print('n_clusters pred: ', len(division))
        print('nmi: ', nmi)
        print('ami: ', ami)
        print('ari: ', ari)

        print('division: ', division)
        # 保存 division 到 CSV 文件
        division_csv_file = f"{folder}/{block}_division.csv"
        save_division_to_csv(division, division_csv_file)

        df.to_csv(f'{folder}/{block}_all_clustered_data.csv', index=False)
        
        
        # 输出错误聚类的数据
        # wrong_clustered_data = []
        # seen_data_points = set()  # 使用集合来跟踪已经添加的数据点

        # for true_label, pred_label in zip(labels_true, prediction):
        #     if true_label!= pred_label:
        #         data_point = df.iloc[pred_label]
        #         # 将数据点转换为元组，并将所有列表转换为元组以确保可哈希性
        #         data_point_tuple = tuple(tuple(item) if isinstance(item, list) else item for item in data_point)
        #         if data_point_tuple not in seen_data_points:
        #             wrong_clustered_data.append(data_point)
        #             seen_data_points.add(data_point_tuple)
        
        # # 将错误聚类的数据保存到文件
        # wrong_clustered_data_df = pd.DataFrame(wrong_clustered_data)
        # wrong_clustered_data_df['prediction'] = prediction  # 添加 prediction 列
        # wrong_clustered_data_df.to_csv(f'{folder}/{block}_wrong_clustered_data.csv', index=False)

           # import networkx as nx
        # import matplotlib.pyplot as plt
        # # # 保存 global_edges 和 stable_points 到 JSON 文件
        # # save_data_json(global_edges, stable_points, folder)

        # # 创建一个图对象
        # G = nx.Graph()
        # # 添加边到图中
        # G.add_edges_from(global_edges)
        # # 打印图的节点和边
        # print("Nodes in the graph:", G.nodes())
        # print("Edges in the graph:", G.edges())

        # # 可视化图
        # nx.draw(G, with_labels=True)
        # plt.savefig(f"{folder}/graph.png")
        # plt.show()  

    return

import csv
def save_division_to_csv(division, filename='division.csv'):
    # Flatten the division list
    rows = []
    for cluster_id, cluster in enumerate(division):
        for node in cluster:
            rows.append([node, cluster_id])  # Store node and its corresponding cluster id
    
    # Create a DataFrame
    df = pd.DataFrame(rows, columns=['Node', 'Cluster_ID'])
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f'Division saved to {filename}')

import json
# 保存 global_edges 和 stable_points 到 JSON 文件
def save_data_json(global_edges, stable_points, save_path):
    with open(f'{save_path}/global_edges.json', 'w') as f:
        json.dump(global_edges, f)
    with open(f'{save_path}/stable_points.json', 'w') as f:
        json.dump(stable_points, f)
def run_hier_2D_SE_mini_Event2012_closed_set(n = 300, e_a = True, e_s = True):
    save_path = './data/Event2012/closed_set/'

    #load test_set_df
    test_set_df_np_path = save_path + 'test_set.npy'
    test_df_np = np.load(test_set_df_np_path, allow_pickle=True)
    test_df = pd.DataFrame(data=test_df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
            "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", 
            "words", "filtered_words", "sampled_words"])
    print("Dataframe loaded.")
    all_node_features = [[str(u)] + \
        [str(each) for each in um] + \
        [h.lower() for h in hs] + \
        e \
        for u, um, hs, e in \
        zip(test_df['user_id'], test_df['user_mentions'],  test_df['hashtags'], test_df['entities'])]

    # load embeddings of the test set messages
    with open(f'{save_path}/SBERT_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    stable_points = get_stable_point(save_path)
    default_num_neighbors = stable_points['first']

    global_edges = get_global_edges(all_node_features, embeddings, default_num_neighbors, e_a = e_a, e_s = e_s)
    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
        if corr_matrix[edge[0]-1, edge[1]-1] > 0] # node encoding starts from 1

    division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n = n)
    prediction = decode(division)

    labels_true = test_df['event_id'].tolist()
    n_clusters = len(list(set(labels_true)))
    print('n_clusters gt: ', n_clusters)

    nmi, ami, ari = evaluate(labels_true, prediction)
    print('n_clusters pred: ', len(division))
    print('nmi: ', nmi)
    print('ami: ', ami)
    print('ari: ', ari)
    return

def run_hier_2D_SE_mini_Event2018_open_set(n = 300, e_a = True, e_s = True, test_with_one_block = True):
    save_path = './data/Event2018/open_set/'
  
    if test_with_one_block:
        blocks = [16]
    else:
        blocks = [i+1 for i in range(16)]
    for block in blocks:
        print('\n\n====================================================')
        print('block: ', block)
        print(datetime.now().strftime("%H:%M:%S"))

        folder = f'{save_path}{block}/'
        
        # load message embeddings
        embeddings_path = folder + 'SBERT_embeddings.pkl'
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        df_np = np.load(f'{folder}{block}.npy', allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=["original_index", "tweet_id", "user_name", "text", "time", "event_id", "user_mentions", \
                "hashtags", "urls", "words", "created_at", "filtered_words", "entities", "sampled_words", "date"])
        all_node_features = [list(set([str(u)] + \
            [str(each) for each in um] + \
            [h.lower() for h in hs] + \
            e)) \
            for u, um, hs, e in \
            zip(df['user_name'], df['user_mentions'],  df['hashtags'], df['entities'])]
        
        stable_points = get_stable_point(folder)
        if e_a == False: # only rely on e_s (semantic-similarity-based edges)
            default_num_neighbors = stable_points['global']
        else:
            default_num_neighbors = stable_points['first']
        if default_num_neighbors == 0: 
            default_num_neighbors = math.ceil((len(embeddings)/1000)*10)
        
        global_edges = get_global_edges(all_node_features, embeddings, default_num_neighbors, e_a = e_a, e_s = e_s)

        corr_matrix = np.corrcoef(embeddings)
        np.fill_diagonal(corr_matrix, 0)
        weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
            if corr_matrix[edge[0]-1, edge[1]-1] > 0] # node encoding starts from 1
        
        division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n = n)
        print(datetime.now().strftime("%H:%M:%S"))

        prediction = decode(division)

        labels_true = df['event_id'].tolist()
        n_clusters = len(list(set(labels_true)))
        print('n_clusters gt: ', n_clusters)

        nmi, ami, ari = evaluate(labels_true, prediction)
        print('n_clusters pred: ', len(division))
        print('nmi: ', nmi)
        print('ami: ', ami)
        print('ari: ', ari)
        
    return

def run_hier_2D_SE_mini_Event2018_closed_set(n = 800, e_a = True, e_s = True):
    save_path = './data/Event2018/closed_set/'

    #load test_set_df
    test_set_df_np_path = save_path + 'test_set.npy'
    test_df_np = np.load(test_set_df_np_path, allow_pickle=True)
    test_df = pd.DataFrame(data=test_df_np, columns=["tweet_id", "user_name", "text", "time", "event_id", "user_mentions", \
            "hashtags", "urls", "words", "created_at", "filtered_words", "entities", "sampled_words"])
    print("Dataframe loaded.")
    all_node_features = [list(set([str(u)] + \
        [str(each) for each in um] + \
        [h.lower() for h in hs] + \
        e)) \
        for u, um, hs, e in \
        zip(test_df['user_name'], test_df['user_mentions'],  test_df['hashtags'], test_df['entities'])]

    # load embeddings of the test set messages
    with open(f'{save_path}/SBERT_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    stable_points = get_stable_point(save_path)
    default_num_neighbors = stable_points['first']

    global_edges = get_global_edges(all_node_features, embeddings, default_num_neighbors, e_a = e_a, e_s = e_s)
    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0]-1, edge[1]-1]) for edge in global_edges \
        if corr_matrix[edge[0]-1, edge[1]-1] > 0] # node encoding starts from 1

    division = hier_2D_SE_mini(weighted_global_edges, len(embeddings), n = n)
    prediction = decode(division)

    labels_true = test_df['event_id'].tolist()
    n_clusters = len(list(set(labels_true)))
    print('n_clusters gt: ', n_clusters)

    nmi, ami, ari = evaluate(labels_true, prediction)
    print('n_clusters pred: ', len(division))
    print('nmi: ', nmi)
    print('ami: ', ami)
    print('ari: ', ari)
    # 找出分类错误的推文索引
    misclassified_indices = [i for i, (true_label, pred_label) in enumerate(zip(labels_true, prediction)) if true_label != pred_label]
    # 输出分类错误的推文
    misclassified_tweets = test_df.iloc[misclassified_indices]
    print("Misclassified tweets:")
    print(misclassified_tweets)
    
    return

if __name__ == "__main__":
    # to run all message blocks, set test_with_one_block to False
    # run_hier_2D_SE_mini_Event2012_open_set(n = 400, e_a = True, e_s = True, test_with_one_block = True)
    #run_hier_2D_SE_mini_Event2012_closed_set(n = 300, e_a = True, e_s = True)
    #run_hier_2D_SE_mini_Event2018_open_set(n = 300, e_a = True, e_s = True, test_with_one_block = True)
    run_hier_2D_SE_mini_Event2018_closed_set(n = 800, e_a = True, e_s = True)
    
