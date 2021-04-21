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
from sklearn.cluster import AgglomerativeClustering
from matplotlib.pylab import show, cm, axis
from matplotlib.colors import LinearSegmentedColormap

seed = 578912
random.seed(seed)
np.random.seed(seed)

def _get_adjacency_matrix(train_path, num_classes, co_occurance_out_file):
    if os.path.exists(co_occurance_out_file):
        objects_adjacency_matrix = np.genfromtxt(co_occurance_out_file, delimiter=',', dtype=float)
    else:
        dir_name = os.path.dirname(train_path)
        print("Extracting the co-occurance matrix from the training dataset")
        objects_adjacency_matrix = np.zeros((num_classes,num_classes))

        # Loop on Images Labels files
        f = open(train_path, "r")
        train_paths = f.readlines()
        train_paths = [
            os.path.join(dir_name, path.rstrip().replace("./", ""). \
                replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt"))
            for path in train_paths
        ]

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

        np.savetxt(co_occurance_out_file, objects_adjacency_matrix, delimiter=",", fmt='%d')

    return objects_adjacency_matrix

def _get_common_classes(adjacency_matrix):
    temp = np.copy(adjacency_matrix)

    temp[temp<100] = 0
    temp[temp>=100] = 1

    common_classes = []
    for i in range(len(temp)):
        if np.sum(temp[i]) >= opt.common_classes_thres:
            common_classes.append(i)
    return common_classes

def _remove_common_classes(objects_adjacency_matrix, common_classes, num_classes):
    # Get mapping to classes idx after removing common classes
    classes_idx_dict = {}
    j = 0
    for i in range(objects_adjacency_matrix.shape[0]):
        if i not in common_classes:
            classes_idx_dict[j] = i
            j += 1

    # Remove common classes
    r_objects_matrix = [[objects_adjacency_matrix[i,j] for j in range(len(objects_adjacency_matrix)) if j not in common_classes] 
                       for i in range(len(objects_adjacency_matrix)) if i not in common_classes]


    r_objects_matrix = np.asarray(r_objects_matrix)
    r_objects_matrix = [[r_objects_matrix[i,j] if i != j else 0 for i in range(len(r_objects_matrix))] for j in range(len(r_objects_matrix)) ]

    r_objects_matrix = np.asarray(r_objects_matrix)

    return classes_idx_dict, r_objects_matrix

def _get_probability_adjacency_matrix(adjacency_matrix):
    prob = np.copy(adjacency_matrix)
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix)):
            prob[i,j] = adjacency_matrix[i,j]/(np.sum(adjacency_matrix[:,j])+1)
    return prob

def _cluster(positions, num_of_clusters):
    keys = sorted([k for k in positions.keys()])
    classes_dict = {}
    for i, key in enumerate(positions.keys()):
        classes_dict[i] = key

    positions = [p for p in positions.values()]
    ag_clusters = AgglomerativeClustering(n_clusters=num_of_clusters).fit(positions)
    labels = ag_clusters.labels_
    clusters = []
    for c in range(num_of_clusters):
        clusters.append([classes_dict[i] for i in range(len(labels)) if labels[i] == c])
    return clusters

def _create_graph(adjacency_matrix, pruning_threshold, itterations):
    ''' create_graph takes an adjacency matrix and the pruning threshold
        It returns the positions of the nodes in the graph,
        and the adjacency matrix in sparse format
    '''
    G = nx.Graph()
    # Add classes names as graph nodes
    for i in range(len(adjacency_matrix)):
        G.add_node(i)

    # Add the a weighed edge between the nodes if 
    # the corresponding value in the adjacency matrix 
    # is greater than the pruning threshold
    for i, row in enumerate(adjacency_matrix):
        for j, w in enumerate(row):
            if i == j:
                continue
            if adjacency_matrix[i,j] > pruning_threshold:
                G.add_edge(i,j,weight=w)

    # Convert the Graph to a sparse format matrix to use in plotting
    s_matrix = nx.to_scipy_sparse_matrix(G)

    # Spead the graph according to the weights to get the nodes positions
    n_positions = nx.spring_layout(G, iterations=itterations)
    
    return n_positions, s_matrix

def _evaluate_clustering(valid_path, common_classes, clusters_list):
    dataset = LoadImagesAndLabels(valid_path, 416, 1, rect=False, single_cls=False, pad=0.5)
    dloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn) # Batch size has to be 1 here

    neglected_objects = 0
    all_objects = 1

    for batch_i, (_, targets, _, _) in enumerate(tqdm(dloader, desc="Evaluating Clusters")):
        ts = targets[:, 1].tolist()
        extras = 0
        cluster_cnt = np.zeros(len(clusters_list))
        for t in ts:
            if t in common_classes:
                extras += 1
                continue
            for i, cluster in enumerate(clusters_list):
                if t in cluster:
                    cluster_cnt[i] += 1
                    break
                
        dominent_clus = [np.argmax(cluster_cnt)] 
        neglected_objects += np.sum(cluster_cnt) - np.sum(cluster_cnt[dominent_clus])
        all_objects += np.sum(cluster_cnt) + extras

    return neglected_objects, all_objects

def _plot_graph(s_adjmatrix, positions, clusters, classes_ids, common_classes, classes_names, fig_name):
    fig = plt.figure()
    graph = nx.Graph(s_adjmatrix)

    # map node to cluster id for colors
    cluster_map = {node: i for i, cluster in enumerate(clusters) for node in cluster}
    colors = [cluster_map[i] for i in range(len(graph.nodes()))]
    
    # retrieve the node labels (class names)
    labels = {}
    j = 0
    for i in range(len(classes_ids)):
        if i not in common_classes:
            labels[j] = classes_names[i]
            j += 1

    nx.draw_networkx_nodes(graph, pos=positions, node_size=700, node_color=colors,cmap=cm.Set3)
    nx.draw_networkx_labels(graph, pos=positions, labels=labels, font_size=12)

    plt.savefig(fig_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--common_classes_thres", type=int, default=55, help="Number of common classes to determine the class as common classes")
    parser.add_argument("--num_clusters", type=int, default=4, help="Number of object clusters")
    parser.add_argument("--data", type=str, default="data/coco2014.data", help="Path to data config file")
    parser.add_argument("--pruning_threshold", type=float, default=0.05, help="Pruning threshold for edge weights")
    parser.add_argument("--co-occurance_out", type=str, default="co-occurence_adjacency_matrix.csv", help="Use if you already have the co-occurance matrix from previous runs")
    parser.add_argument('--eval', action='store_true', help='Evaluate the clustering')

    opt = parser.parse_args()

    data_config = parse_data_cfg(opt.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    num_classes = int(data_config["classes"])
    classes_names = load_classes(data_config['names'])

    num_clusters = opt.num_clusters
    dir_name = os.path.dirname(train_path)

    ##### Compute the frequency based adjacency Matrix #####
    adjacency_matrix = _get_adjacency_matrix(train_path=train_path, 
                                            num_classes=num_classes, 
                                            co_occurance_out_file=opt.co_occurance_out)

    chosen_classes = [0, 2, 3, 5, 7, 9, 10, 11, 12, 18, 17, 19, 26, 22, 41, 42, 43, 44, 
                      45, 48, 57, 63, 62, 64, 65, 66, 68, 69, 71, 73]
    print("Chosen subset:", len(chosen_classes))
    # chosen_classes = np.arange(80)
    classes_names = np.asarray(classes_names)

    if len(chosen_classes) < num_classes:
        classes_names = classes_names[chosen_classes]
        adjacency_matrix = adjacency_matrix[chosen_classes,:]
        adjacency_matrix = adjacency_matrix[:,chosen_classes]

    # Get Common Classes
    common_classes = _get_common_classes(adjacency_matrix)
    print("Number of common_classes = ", len(common_classes), classes_names[common_classes])
    classes_idx_dict, adjacency_matrix = _remove_common_classes(adjacency_matrix, 
                                                                common_classes=common_classes,
                                                                num_classes=num_classes)

    # Convert frequency based agjacency matrix to probablility based adjacency matrix
    prob_adjacency_matrix = _get_probability_adjacency_matrix(adjacency_matrix)
            
    # Convert the probability based adjacency matrix to a graph
    n_positions, s_matrix = _create_graph(prob_adjacency_matrix, pruning_threshold=opt.pruning_threshold, itterations=200)

    # Cluster the classes based on their positions in the graph
    clusters = _cluster(n_positions, num_clusters)

    # Plot the graph with the clustered classes

    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("output/clustering"):
        os.makedirs("output/clustering")

    fig_name = "output/clustering/cluster_" + str(opt.num_clusters) + "_" + str(opt.common_classes_thres) 
    _plot_graph(s_matrix, n_positions, clusters, chosen_classes, common_classes, classes_names, fig_name + ".png")

    ## Add common classes to clusters list and save the clusters file
    for idx, cluster in enumerate(clusters):
        clusters[idx] = [classes_idx_dict[i] for i in cluster]
        clusters[idx].extend(common_classes)
    
    with open(fig_name + ".data", 'w') as f:
        for cluster in clusters:
            for i, obj in enumerate(cluster):
                if i == 0:
                    f.write("%s"%(obj))
                else:
                    f.write(",%s"%(obj))
            f.write("\n")

    ## Evaluate the clustering by checking the branch miss rate
    if opt.eval:
        neglected, all_obj = _evaluate_clustering(valid_path, common_classes, clusters)
        print(f'Separability Error = {neglected/all_obj*100}%')


