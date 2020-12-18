from __future__ import division

from utils.datasets import *
from utils.parse_config import parse_data_cfg
from utils.utils import *

import sys
import argparse

from torch.utils.data import DataLoader
import random

import os
import numpy as np 
import matplotlib.pyplot as plt

import networkx as nx
from sklearn.cluster import KMeans,AgglomerativeClustering
from matplotlib.pylab import show, cm, axis
from matplotlib.colors import LinearSegmentedColormap

seed = 578912
random.seed(seed)
np.random.seed(seed)

def kmeans_clustering(classes_positions, num_of_clusters):
    keys = sorted([k for k in classes_positions.keys()])
    classes_dict = {}
    for i, key in enumerate(classes_positions.keys()):
        classes_dict[i] = key

    positions = [p for p in classes_positions.values()]
    #kmeans = KMeans(n_clusters=num_of_clusters, random_state=0,n_init=100).fit(positions)
    kmeans = AgglomerativeClustering(n_clusters=num_of_clusters).fit(positions)
    labels = kmeans.labels_
    clusters = []
    for c in range(num_of_clusters):
        clusters.append([classes_dict[i] for i in range(len(labels)) if labels[i] == c])
    return clusters

def evaluate_clustering(valid_path, common_classes, clusters_list):
    dataset = LoadImagesAndLabels(valid_path, 416, 8, rect=False, single_cls=False, pad=0.5)
    dloader = DataLoader(dataset,
                        batch_size=8,
                        num_workers=1,
                        pin_memory=True,
                        collate_fn=dataset.collate_fn)

    neglected_objects = 0
    all_objects = 1

    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dloader, desc="Evaluating Clusters")):
        ts = targets[:, 1].tolist()
        extras = 0
        cluster_cnt = np.zeros(len(clusters_list))
        for t in ts:
            for i, cluster in enumerate(clusters_list):
                if t in cluster and t not in common_classes:
                    cluster_cnt[i] += 1
                if t in common_classes:
                    extras += 1
        # dominent_clus = [idx for idx, val in enumerate(cluster_cnt) if val != 0] 
        dominent_clus = [np.argmax(cluster_cnt)] 
        neglected_objects += np.sum(cluster_cnt) - np.sum(cluster_cnt[dominent_clus])
        all_objects += np.sum(cluster_cnt) + extras

    return neglected_objects, all_objects

def create_graph(matrix,pruning_threshold):
    G = nx.Graph()
    for i in range(len(matrix)):
        G.add_node(i)
    for i, row in enumerate(matrix):
        for j, w in enumerate(row):
            if i == j:
                continue
            if matrix[i,j] > pruning_threshold:
                G.add_edge(i,j,weight=w)

    mat = nx.to_scipy_sparse_matrix(G)
    
    return G, mat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--common_thresh", type=int, default=80, help="Number of classes")
    parser.add_argument("--num_clusters", type=int, default=4, help="Number of object clusters")
    parser.add_argument("--output", type=str, default="cluster_gen.data", help="Path to output the clusters")
    parser.add_argument("--data", type=str, default="data/coco2014.data", help="path to data config file")
    parser.add_argument("--pruning_threshold", type=float, default=0.01, help="Pruning threshold for edge weights")
    parser.add_argument("--cooccurance", type=str, default="co-occurence_adjacency_matrix.csv", help="Use if you already have the co-occurance matrix from previous runs")

    opt = parser.parse_args()

    data_config = parse_data_cfg(opt.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    num_classes = int(data_config["classes"])
    classes_names = load_classes(data_config['names'])

    num_clusters = opt.num_clusters

    ##### Compute the frequency based adjacency Matrix #####
    if not os.path.exists(opt.cooccurance):
        print("Extracting the co-occurance matrix from the training dataset")
        objects_adjacency_matrix = np.zeros((num_classes,num_classes))

        # Loop on Images Labels files
        f=open(train_path, "r")
        train_paths = f.readlines()
        train_paths = [path.rstrip().replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
                for path in train_paths]

        for filename in train_paths:
            if filename.endswith(".txt"):
                # Loop on Objects inside each Image
                objects = []
                with open(filename, "r") as a_file:
                    for line in a_file:
                        stripped_line = line.split(" ")
                        objects.append(int(stripped_line[0]))
                for obj1 in objects:
                    for obj2 in objects:
                        objects_adjacency_matrix[obj1,obj2] += 1
                        objects_adjacency_matrix[obj2,obj1] += 1

        np.savetxt("co-occurence_adjacency_matrix.csv", objects_adjacency_matrix, delimiter=",", fmt='%d')

    else:
        objects_adjacency_matrix = np.genfromtxt(opt.cooccurance, delimiter=',',dtype=float)

    # chosen_classes = [0, 2,3, 5, 7,9,12, 18, 17, 19, 26,  41, 42, 43, 44, 45, 48, 57, 63, 62, 64,65,66,68,69, 71, 73]
    chosen_classes = np.arange(80)
    classes_names = np.asarray(classes_names)
    classes_names = classes_names[chosen_classes]

    #### Cluster the objects according to the adjacency matrix #####
    objects_adjacency_matrix = objects_adjacency_matrix[chosen_classes,:]
    objects_adjacency_matrix = objects_adjacency_matrix[:,chosen_classes]

    # Get Common Classes
    temp = np.copy(objects_adjacency_matrix)

    temp[temp<100] = 0
    temp[temp>=100] = 1

    common_count = 0
    common_classes = []
    for i in range(len(temp)):
        if np.sum(temp[i]) >= opt.common_thresh:
            common_classes.append(i)
            common_count += 1

    # Get mapping to classes idx after removing common classes
    classes_idx_dict = {}
    j = 0
    for i in range(num_classes):
        if i not in common_classes:
            classes_idx_dict[j] = i
            j += 1

    # Remove common classes
    r_objects_matrix = [[objects_adjacency_matrix[i,j] for j in range(len(objects_adjacency_matrix)) if j not in common_classes] 
                       for i in range(len(objects_adjacency_matrix)) if i not in common_classes]


    r_objects_matrix = np.asarray(r_objects_matrix)
    r_objects_matrix = [[r_objects_matrix[i,j] if i != j else 0 for i in range(len(r_objects_matrix))] for j in range(len(r_objects_matrix)) ]

    r_objects_matrix = np.asarray(r_objects_matrix)

    # Get the probabilities for the adjacency matrix
    prob = np.copy(r_objects_matrix)
    for i in range(len(r_objects_matrix)):
        for j in range(len(r_objects_matrix)):
            prob[i,j] = r_objects_matrix[i,j]/(np.sum(r_objects_matrix[:,j])+1)

    ## Draw Graph and MCL Clustering results
    labels = {}
    j = 0
    for i in range(len(chosen_classes)):
        if i not in common_classes:
            labels[j] = classes_names[i]
            j += 1
            
    G, mat = create_graph(prob,0.005)
    pos=nx.spring_layout(G,iterations=40)

    clusters = kmeans_clustering(pos, num_clusters)

    # make a networkx graph from the adjacency matrix
    fig = plt.figure()
    graph = nx.Graph(mat)

    # map node to cluster id for colors
    cluster_map = {node: i for i, cluster in enumerate(clusters) for node in cluster}
    colors = [cluster_map[i] for i in range(len(graph.nodes()))]
    
    # draw
    nx.draw_networkx_nodes(graph,  pos=pos, node_size=700, node_color=colors,cmap=cm.cool)
    nx.draw_networkx_labels(graph,  pos=pos, labels=labels, font_size=12)

    plt.show()

    ## Add common classes to clusters list
    for idx, cluster in enumerate(clusters):
        clusters[idx] = [classes_idx_dict[i] for i in cluster]
        clusters[idx].extend(common_classes)
    
    for c in clusters:
        print(len(c), classes_names[c])
    
    with open(opt.output, 'w') as f:
        for cluster in clusters:
            for obj in cluster:
                f.write("%s,"%(obj))
            f.write("\n")

    ## Evaluate the clustering by checking the branch miss rate
    neglected, all_obj = evaluate_clustering(valid_path, common_classes,clusters)
    print(neglected/all_obj*100, "%")


