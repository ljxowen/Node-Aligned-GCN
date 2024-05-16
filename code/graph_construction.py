import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import torch

import codebook_generation
import utils
from tqdm import tqdm


def graph_construct(all_graph_data, selected_patient_data_map, miss_node_map, inner_cluster_k):
    print("Constructing Graph........")
    adj_matrix_data = {}
    all_graph_data_np = np.array(all_graph_data)
    # create the adj matrix for each patient(wsi)
    for patient, indices in tqdm(selected_patient_data_map.items()):
        # read the each graph data by patient name
        graph_data = all_graph_data_np[indices]
        
        # get current miss node
        miss_node = miss_node_map[patient]
        
        # createt the adjacent matrix
        feature_for_dis = graph_data
        # calculate the euclidean distance betwee each instance
        fea_distances = euclidean_distances(feature_for_dis)
        adj_matrix = np.zeros((len(feature_for_dis), len(feature_for_dis)))
        max_adge_num = 3

        for i, distance in enumerate(fea_distances):
            sub_bag = i // inner_cluster_k # get the current inner cluster label
            if np.all(feature_for_dis[i, :] == 0):# ith feature is empty
                continue
            if sub_bag in miss_node: # if sub bag is in miss node
                continue
            dis_idx = [[dis, j] for j, dis in enumerate(distance)] 
            dis_idx.sort() # sort the list of distance between i and each j
            count = 0

            for dis, j in dis_idx:
                cur_sub_bag = j // inner_cluster_k

                if np.all(feature_for_dis[j, :] == 0): continue
                if sub_bag in miss_node: continue

                if cur_sub_bag == sub_bag:
                    adj_matrix[i][j] = 1
                else:
                    if count >= max_adge_num:
                        continue
                    else:
                        adj_matrix[i][j] = 1
                        count += 1
        # store the adj matrix with patient id
        adj_matrix_data[patient] = adj_matrix
    
    return adj_matrix_data



if __name__ == '__main__':
    # Testing
    path = "E:/gnn_project/datasets"
    data, IDC_label, patients = utils.load_data(path)
    patient_data_map = utils.create_patient_data_map(patients)
    target_label_map = utils.create_target_label(patient_data_map, IDC_label)

    img_features = torch.load('img_features.pt')

    inner_cluster_k = 25
    labels, pca_data, global_cluster_num = codebook_generation.global_cluster(100, img_features, 0.3, False)
    all_graph_data, selected_patient_data_map, selected_target_label, miss_node_map = codebook_generation.inner_cluster(labels, pca_data,
                                                                                                         img_features, patient_data_map,
                                                                                                         target_label_map, inner_cluster_k)

    adj_matrix_data = graph_construct(all_graph_data, selected_patient_data_map, miss_node_map, inner_cluster_k)

    print(f"Number of adjacent matrix: {len(adj_matrix_data)}")
    print(f"Numebr of target label: {len(selected_target_label)}")

