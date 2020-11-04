from __future__ import division

from models.ada_models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import random

import os
import argparse
import numpy as np 
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import networkx as nx

from sklearn.cluster import KMeans
import markov_clustering as mc

def kmeans_clustering(classes_positions, num_of_clusters):
    keys = sorted([k for k in classes_positions.keys()])
    print(len(keys), keys)
    classes_dict = {}
    for i, key in enumerate(classes_positions.keys()):
        classes_dict[i] = key

    positions = [p for p in classes_positions.values()]
    kmeans = KMeans(n_clusters=num_of_clusters, random_state=0,n_init=100).fit(positions)
    labels = kmeans.labels_
    clusters = []
    for c in range(num_of_clusters):
        clusters.append([classes_dict[i] for i in range(len(labels)) if labels[i] == c])
    print("clustersss ",clusters)
    return clusters

def evaluate_clustering(valid_path, common_classes, clusters_list):
    dataset = ListDataset(valid_path, img_size=416, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    neglected_objects = 0
    all_objects = 1

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Evaluating clusters")):
        ts = targets[:, 1].tolist()
        cluster_cnt = np.zeros(len(clusters_list))
        for t in ts:
            for i, cluster in enumerate(clusters_list):
                #if t in cluster and t not in common_classes:
                    cluster_cnt[i] += 1
        # dominent_clus = [idx for idx, val in enumerate(cluster_cnt) if val != 0] 
        dominent_clus = [np.argmax(cluster_cnt)] 
        neglected_objects += np.sum(cluster_cnt) - np.sum(cluster_cnt[dominent_clus])
        all_objects += np.sum(cluster_cnt)

    return neglected_objects, all_objects

def balance_clusters(clusters_list, min_objects_per_cluster=8):
    final_clusters = []
    temp_cluster = []
    for cluster in clusters_list:
        if len(cluster) < min_objects_per_cluster:
            if len(temp_cluster) > int(70/4):
                final_clusters.append(temp_cluster)
                temp_cluster = []
            
            temp_cluster.extend([i for i in cluster])
        else:
            final_clusters.append([i for i in cluster])
    final_clusters.append(temp_cluster)

    return final_clusters

def get_weighted_positions(matrix):
    G = nx.MultiDiGraph()
    for i, row in enumerate(matrix):
        for j, w in enumerate(row):
            if i == j:
                continue
            if matrix[i,j] > 0.01:
                G.add_edge(i,j,weight=w)
                G.add_edge(j,i,weight=w)
    
    pos=nx.spring_layout(G)
    nx.draw_networkx(G, pos=pos)
    plt.show()
    return G, pos

def create_graph(matrix,pruning_threshold):
    G = nx.MultiDiGraph()
    for i, row in enumerate(matrix):
        for j, w in enumerate(row):
            if i == j:
                continue
            if matrix[i,j] > pruning_threshold:
                G.add_edge(i,j,weight=w)
                G.add_edge(j,i,weight=w)

    mat = nx.to_scipy_sparse_matrix(G)
    
    return G, mat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default='data/coco/val2014/', help="Path of labels directory")
    parser.add_argument("--num_classes", type=int, default=80, help="Number of classes")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of object clusters")
    parser.add_argument("--output", type=str, default="cluster.data", help="Path to output the clusters")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--kmeans", type=bool, default=False, help="Do kmeans clustering, default is MCL")
    parser.add_argument("--pruning_threshold", type=float, default=0.01, help="Pruning threshold for edge weights")

    opt = parser.parse_args()

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]

    directory = opt.labels
    num_classes = opt.num_classes
    num_clusters = opt.num_clusters

    ##### Compute the frequency based adjacency Matrix #####

    # objects_adjacency_matrix = np.zeros((num_classes,num_classes))

    # # Loop on Images Labels files
    # for filename in os.listdir(directory):
    #     if filename.endswith(".txt"):
    #         # Loop on Objects inside each Image
    #         objects = []
    #         with open(directory + filename, "r") as a_file:
    #             for line in a_file:
    #                 stripped_line = line.split(" ")
    #                 objects.append(int(stripped_line[0]))
    #         for obj1 in objects:
    #             for obj2 in objects:
    #                 objects_adjacency_matrix[obj1,obj2] += 1
    #                 objects_adjacency_matrix[obj2,obj1] += 1

    # np.savetxt("co-occurence_adjacency_matrix.csv", objects_adjacency_matrix, delimiter=",", fmt='%d')
    
    objects_adjacency_matrix = np.genfromtxt('co-occurence_adjacency_matrix.csv', delimiter=',',dtype=float)

    #### Cluster the objects according to the adjacency matrix #####

    classes_names = []
    with open("data/coco.names", "r") as file_names:
        for line in file_names:
            classes_names.append(line[:-1])
    classes_names = np.asarray(classes_names)

    # Get Common Classes
    temp = np.copy(objects_adjacency_matrix)

    temp[temp<50] = 0
    temp[temp>=50] = 1

    common_count = 0
    common_classes = []
    for i in range(len(temp)):
        if np.sum(temp[i]) >= 60:
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
    
    # Get the probabilities for the adjacency matrix
    prob = np.copy(r_objects_matrix)
    for i in range(len(r_objects_matrix)):
        for j in range(len(r_objects_matrix)):
            prob[i,j] = r_objects_matrix[i,j]/(np.sum(r_objects_matrix[:,j])+1)

    ## Draw Graph and MCL Clustering results
    labels = {}
    j = 0
    for i in range(len(classes_names)):
        if i not in common_classes:
            labels[j] = classes_names[i]
            j += 1
            
    ## Repeat clustering for 
    if opt.kmeans == True:
        print("Seed", np.random.seed)
        G, mat = create_graph(prob,0.01)
        pos=nx.spring_layout(G,iterations=40)

        clusters = kmeans_clustering(pos, 4)
        print("Kmeans Clusters", clusters)
        mc.draw_graph(mat, clusters, pos=pos, labels=labels, font_size=8, node_size=1000)
        plt.show()
    else:
        G, mat = create_graph(prob,opt.pruning_threshold)

        # Do MCL
        result = mc.run_mcl(mat, pruning_threshold=opt.pruning_threshold, pruning_frequency=5, iterations=100, inflation=1.4, expansion=3)   # run MCL with default parameters
        clusters = mc.get_clusters(result)    # get clusters

        assert len(common_classes) + sum([len(cluster) for cluster in clusters]) == num_classes

        pos=nx.spring_layout(nx.Graph(mat),iterations=80)

        mc.draw_graph(mat, clusters, pos=pos, labels=labels, font_size=8, node_size=1000)
        clusters = balance_clusters(clusters,7)
        plt.show()

    assert len(common_classes) + sum([len(cluster) for cluster in clusters]) == num_classes

    print("Common classes", len(common_classes))

    ## Add common classes to clusters list
    for idx, cluster in enumerate(clusters):
        clusters[idx] = [classes_idx_dict[i] for i in cluster]
        clusters[idx].extend(common_classes)
        #print(idx, len(clusters[idx]))
    #clusters.append(common_classes)
    print(clusters)

    ## Evaluate the clustering by checking the branch miss rate
    neglected, all_obj = evaluate_clustering(valid_path, common_classes,clusters)
    print(neglected/all_obj)

    with open("clusters_gen1.data", 'w') as f:
        for cluster in clusters:
            for obj in cluster:
                f.write(",%s"%(obj))
            f.write("\n")
