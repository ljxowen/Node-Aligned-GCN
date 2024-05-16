import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import torch

import warnings
from tqdm import tqdm
from collections import Counter

import utils
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")   



def global_cluster(global_cluster_num, img_features, train_ratio = -1, opt_k = False, k_max = 10, log = False):
    print("Generate codebook...............")
    # Clustering
    global_cluster_data = img_features.numpy()

    # using pca to decrease feature dimension
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(global_cluster_data) # pca data for cluster

    # prepare the training data
    if train_ratio > 0 and train_ratio <= 1:
        num_selections = int(pca_data.shape[0] * train_ratio)
        random_indices = np.random.choice(pca_data.shape[0], num_selections, replace=False) 
        train_pca_data = pca_data[random_indices]
    else:
        train_pca_data = pca_data

    print(f"Train data: {len(train_pca_data)}")
    print(f"Total data: {len(pca_data)}")

    # find optimal k for cluster
    if opt_k == True:
        print("Find optimal K.......")
        silhouette_coefficients = {}
        for k in tqdm(range(3, k_max)): 
            clusterer = KMeans(n_clusters=k)
            #clusterer = MiniBatchKMeans(n_clusters=k, batch_size=10000)
            clusterer.fit(train_pca_data)
            score = silhouette_score(train_pca_data, clusterer.labels_)
            silhouette_coefficients[k] = score
        best_k = max(silhouette_coefficients, key=silhouette_coefficients.get)
    
    # global cluster
    if opt_k == True:
        global_cluster_num = best_k
        clusterer = KMeans(n_clusters=global_cluster_num)
    else:
        clusterer = KMeans(n_clusters=global_cluster_num)
        
    if train_ratio > 0 and train_ratio <= 1:
        clusterer.fit(train_pca_data) # only fit the training data
        labels = clusterer.predict(pca_data)
    else:
        labels = clusterer.fit_predict(pca_data)

    # count the label in cluster
    if log == True:
        cluster_counts = Counter(labels)
        for cluster_label, count in cluster_counts.items():
            print(f"Cluster {cluster_label} has {count} data points.")

    return labels, pca_data, global_cluster_num



def inner_cluster(labels, pca_data, img_features, patient_data_map, target_label_map, inner_cluster_k):
    print("inner clustering.......")
    all_graph_data = [] # selected img_features for graph construction
    global_cluster_data = img_features.numpy()
    fea_dim = len(global_cluster_data[0])
    selected_patient_data_map = {} # init the indices of patient id

    # inner cluster on each global label of each wsi
    unique_label = np.unique(labels)

    miss_node_map = {patient: set([i for i in range(len(unique_label))]) for patient in patient_data_map.keys()} # track of clusters without sufficient data
    indx_count = 0
    for patient, indices in patient_data_map.items():
        selected_patient_data_map[patient] = [] # init the indices of patient id
        for label in unique_label:
            # for each cluster in global
            bool_indices = np.where(labels == label)[0] #original global index
            curr_indices = np.isin(bool_indices, indices) #bool indcies also in current patient
            bool_indices =bool_indices[curr_indices]

            if len(bool_indices) >= inner_cluster_k:
                miss_node_map[patient].remove(label) # remove current label, since data is valid
            
                inner_pca_data = pca_data[bool_indices] #pca data for cluster
                inner_cluster_data = global_cluster_data[bool_indices]# original feature data for inner cluster
                
                # perform the inner cluster
                inner_clusterer = KMeans(n_clusters=inner_cluster_k)
                #inner_clusterer = MiniBatchKMeans(n_clusters=inner_cluster_k, batch_size=10000)
                inner_clusterer.fit(inner_pca_data)
                inner_labels = inner_clusterer.labels_
                
                # randomly select fixed num individual in each inner cluster
                for inner_label in range(inner_cluster_k):
                    inner_indices = np.where(inner_labels == inner_label)[0]
                    random_idx = np.random.choice(inner_indices)
                    selected_ind = inner_cluster_data[random_idx] #selected individual by inner cluster
                    all_graph_data.append(selected_ind) # selected img feature for graph construction
                    selected_patient_data_map[patient].append(indx_count)
                    indx_count += 1
            else: # hand embed zero feature 
                for j in range(inner_cluster_k):
                    zero_ind = np.zeros(fea_dim)
                    all_graph_data.append(zero_ind)
                    selected_patient_data_map[patient].append(indx_count)
                    indx_count += 1

    #update the target label map for selected sample
    selected_target_label = []
    for patient in selected_patient_data_map.keys():
        selected_target_label.append(target_label_map[patient])
     
    print(f"The original number of global individuals: {len(global_cluster_data)}")
    print(f"The number after inner cluster selected: {len(all_graph_data)}")

    return all_graph_data, selected_patient_data_map, selected_target_label, miss_node_map



if __name__ == '__main__':
    # Testing
    path = "E:/gnn_project/datasets"
    data, IDC_label, patients = utils.load_data(path)
    patient_data_map = utils.create_patient_data_map(patients)
    target_label_map = utils.create_target_label(patient_data_map, IDC_label)
    #img_features, patient_data_map, target_label_map = data_encode.data_encoder(path)

    img_features = torch.load('img_features.pt')
    labels, pca_data, global_cluster_num = global_cluster(50, img_features, 0.5, False)

    # visualization the global cluster
    plt.figure(figsize=(10, 6))
    for label in np.unique(labels):
        # get the data point (2d) for each label
        c_data = pca_data[labels == label]
        # plot the data point [x,y]
        plt.scatter(c_data[:, 0], c_data[:, 1], label=f'Cluster {label}')
    plt.title('PCA-reduced Data Clustered')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    #plt.legend()
    plt.show()

    inner_cluster_k = 10
    all_graph_data, selected_patient_data_map, selected_target_label, _ = inner_cluster(labels, pca_data,
                                                                                     img_features, patient_data_map,
                                                                                     target_label_map, inner_cluster_k)
