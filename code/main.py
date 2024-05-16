import codebook_generation
import graph_construction
import data_encode
import train
import utils

import torch
import argparse
import pickle
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--fea_dim', type=int, default=1024)
    parser.add_argument('--class_num', type=int, default=1)
    parser.add_argument('--global_cluster_num', type=int, default=50)
    parser.add_argument('--inner_cluster_num', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--lr',type=float, default=5e-5)
    parser.add_argument('--L2_reg', type=float, default=5e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    
    args = parser.parse_args()
    path = "E:/gnn_project/datasets"
    encode_fea_preload = False

    if encode_fea_preload == False:
        img_features, patient_data_map, target_label_map = data_encode.data_encoder(path)
        torch.save(img_features, 'img_features.pt') #save img_feature
        print("encode img_features saved")
    else:
        data, IDC_label, patients = utils.load_data(path)
        patient_data_map = utils.create_patient_data_map(patients)
        target_label_map = utils.create_target_label(patient_data_map, IDC_label)
        img_features = torch.load('img_features.pt')
    
    inner_cluster_k = args.inner_cluster_num
    labels, pca_data, global_cluster_num = codebook_generation.global_cluster(args.global_cluster_num, img_features, 0.5, opt_k = False)
    all_graph_data, selected_patient_data_map, selected_target_label, miss_node_map = codebook_generation.inner_cluster(labels, pca_data, img_features,
                                                                                                          patient_data_map, 
                                                                                                          target_label_map, inner_cluster_k)

    adj_matrix_data = graph_construction.graph_construct(all_graph_data, selected_patient_data_map, miss_node_map, inner_cluster_k)

    train_loss_log, train_acc_log, valid_acc_log = train.main(args, adj_matrix_data, all_graph_data,
                                                              selected_patient_data_map, selected_target_label,
                                                              miss_node_map,
                                                              inner_cluster_k, global_cluster_num)
    
    # save the model logs
    logs = {
        "train_loss_log": train_loss_log,
        "train_acc_log": train_acc_log,
        "valid_acc_log": valid_acc_log
    }

    current_dir = os.path.dirname(__file__)
    file_name = "training_logs.pkl"
    file_path = os.path.join(current_dir, file_name)

    with open(file_path, "wb") as file:
        pickle.dump(logs, file)

    # open tha log file
    #with open(file_path, "rb") as file:
    #   loaded_logs = pickle.load(file)