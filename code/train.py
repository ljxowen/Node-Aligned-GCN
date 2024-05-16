import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split

import numpy as np

import codebook_generation
import graph_construction
import data_encode
import gcn_model


class NAGCN_Dataset(Dataset):
    def __init__(self, adj_matrix_data, all_graph_data, selected_patient_data_map, selected_target_label, miss_node_map, inner_cluster_k, global_cluster_k):
        self.node_nums = inner_cluster_k * global_cluster_k
        self.groups = torch.tensor([i // inner_cluster_k for i in range(self.node_nums)]).float()
        self.inner_cluster_num = inner_cluster_k
        self.global_cluster_k = global_cluster_k
        self.adj_matrix_data = list(adj_matrix_data.values()) # this rely on python version > 3.7 to keep order of data
        self.all_graph_data = np.array(all_graph_data)
        self.selected_patient_list = list(selected_patient_data_map.keys())
        self.selected_patient_data_map = selected_patient_data_map
        self.selected_target_label = selected_target_label
        self.miss_node_map = miss_node_map

    def __len__(self):
        return len(self.selected_patient_list)

    def __getitem__(self, idx):
        # wsi features
        curr_patient = self.selected_patient_list[idx]
        curr_indices = self.selected_patient_data_map[curr_patient]
        curr_img_feature = self.all_graph_data[curr_indices]
        curr_img_feature = torch.from_numpy(curr_img_feature)
        curr_img_feature = curr_img_feature.float()

        # adjacent matrix
        curr_adj_matrix = self.adj_matrix_data[idx]
        curr_adj_matrix = torch.from_numpy(curr_adj_matrix)
        curr_adj_matrix = curr_adj_matrix.float()

        # mask matrix (for optimize weight of node)
        curr_mask_matrix = np.ones(self.global_cluster_k)
        curr_miss_node = self.miss_node_map[curr_patient]
        for label in curr_miss_node:
            curr_mask_matrix[label] = 0
        curr_mask_matrix = torch.from_numpy(curr_mask_matrix)
        curr_mask_matrix = curr_mask_matrix.float()

        # target label
        curr_target_label = self.selected_target_label[idx]
        curr_target_label = torch.tensor(curr_target_label)

        groups = self.groups

        inner_node_num = self.inner_cluster_num

        return curr_img_feature, curr_target_label, curr_adj_matrix, curr_mask_matrix, inner_node_num, groups
    


def train(train_dataloader, model, optimizer, criterion, log_interval = 100, batch_log = False):
    model.train() #set model for training

    total_loss = 0
    correct_pred = 0
    total_samples = 0

    for batch_indx, (x_feature, y_label, adj_matrix, mask_matrix, inner_node_n, groups) in enumerate(train_dataloader):
        # load data to gpu
        x_feature = x_feature.to('cuda')
        y_label = y_label.to('cuda')
        adj_matrix = adj_matrix.to('cuda')
        mask_matrix = mask_matrix.to('cuda')
        inner_node_n = inner_node_n.to('cuda')
        groups = groups.to('cuda')

        # clean the gradient
        optimizer.zero_grad()

        # model output
        output = model(x_feature, adj_matrix, mask_matrix, groups)

        # loss function calculation (CrossEntropyLoss)
        y_label = y_label.unsqueeze(1).float()
        loss = criterion(output, y_label)

        # compute gradient
        loss.backward()

        # update parameter
        optimizer.step()

        # compute metrics each batch
        #predicts = torch.argmax(output, dim = 1) # select highest possibility as predict result (for mutlti classification)
        predicts = (output > 0.5).float()
        correct_pred += (predicts == y_label).sum().item() # correct predict number
        total_samples += y_label.size(0) # number of samples in this batch
        total_loss += loss.item()

        # print log for each batch
        if batch_indx % log_interval == 0 and batch_log == True:
            print(f"Batch: {batch_indx}/{len(train_dataloader)} | Loss: {loss.item():.4f}")

    # total metrics
    avg_loss = total_loss / len(train_dataloader)
    train_accuracy = correct_pred / total_samples
    # print log
    if batch_log == True:
        print(f'Train Loss: {avg_loss:.4f} | Train Accuracy: {train_accuracy:.4f}')

    return avg_loss, train_accuracy



def eval(test_dataloader, model, criterion, log_interval = 100, batch_log = False):
    model.eval() # set model for evaluate

    total_loss = 0
    correct_predict = 0
    total_samples = 0

    # evaluate model
    with torch.no_grad():
        for batch_indx, (x_feature, y_label, adj_matrix, mask_matrix, inner_node_n, groups) in enumerate(test_dataloader):
            x_feature = x_feature.to('cuda')
            y_label = y_label.to('cuda')
            adj_matrix = adj_matrix.to('cuda')
            mask_matrix = mask_matrix.to('cuda')
            inner_node_n = inner_node_n.to('cuda')
            groups = groups.to('cuda')

            output = model(x_feature, adj_matrix, mask_matrix, groups)

            y_label = y_label.unsqueeze(1).float()
            loss = criterion(output, y_label)
            total_loss += loss.item()

            #predicts = torch.argmax(output, dim = 1) 
            predicts = (output > 0.5).float()

            correct_predict += (predicts == y_label).sum().item()
            total_samples += y_label.size(0)

            if batch_indx % log_interval == 0 and batch_log == True:
                print(f"Batch: {batch_indx + 1}/{len(test_dataloader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(test_dataloader)
    eval_accuracy = correct_predict / total_samples

    if batch_log == True:
        print(f'Eval Loss: {avg_loss:.4f} | Eval Accuracy: {eval_accuracy:.4f}')

    return avg_loss, eval_accuracy



def main(args, adj_matrix_data, all_graph_data, selected_patient_data_map, selected_target_label, miss_node_map, inner_cluster_k, global_cluster_k):
    #create the dataset
    print("Prepare dataset.......")
    dataset = NAGCN_Dataset(adj_matrix_data, all_graph_data, selected_patient_data_map, 
                            selected_target_label, miss_node_map, inner_cluster_k, global_cluster_k)

    # split train and test dataset
    data_size = len(dataset)
    train_size = int(data_size * 0.8)
    test_size = data_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    #create data loader for model training and evaluation
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = False, num_workers = 0)
    test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False, num_workers = 0)

    print("Training model.......")
    # set gcn model
    model = gcn_model.GCN(fea_dim = args.fea_dim, class_num = args.class_num, 
                          global_cluster_num = args.global_cluster_num, dropout = args.dropout)
    model = model.cuda()

    # set optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2_reg)
    #criterion = torch.nn.CrossEntropyLoss() #(for muti class)
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()

    # metrics
    train_loss_log = []
    train_acc_log = []
    valid_acc_log = []

    # training and validation
    for epoch in range(args.num_epochs):
        train_loss, train_accuracy = train(train_dataloader, model, optimizer, criterion)
        _, test_accuracy = eval(test_dataloader, model, criterion)

        # logging
        print(f'Epoch: {epoch+1}/{args.num_epochs} ' 
              f'| Train Loss: {train_loss:.4f} '
              f'| Train Acc: {train_accuracy:.4f} '
              f'| Valid Acc: {test_accuracy:.4f}')
        
        # save the log history
        train_loss_log.append(train_loss)
        train_acc_log.append(train_accuracy)
        valid_acc_log.append(test_accuracy)
        
    return train_loss_log, train_acc_log, valid_acc_log


if __name__ == '__main__':
    # Testing
    path = "E:/gnn_project/datasets"
    img_features, patient_data_map, target_label_map = data_encode.data_encoder(path)

    inner_cluster_k = 15
    global_cluster_k = 4
    labels, pca_data, _ = codebook_generation.global_cluster(global_cluster_k, img_features, 0.3, False)
    all_graph_data, selected_patient_data_map, selected_target_label, miss_node_map = codebook_generation.inner_cluster(labels, pca_data, img_features,
                                                                                                          patient_data_map,
                                                                                                        target_label_map, inner_cluster_k)

    adj_matrix_data = graph_construction.graph_construct(all_graph_data, selected_patient_data_map, miss_node_map, inner_cluster_k)

    #create the dataset and the data loader for model
    dataset = NAGCN_Dataset(adj_matrix_data, all_graph_data, selected_patient_data_map, selected_target_label, miss_node_map, inner_cluster_k, 4)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = False, num_workers = 0)
    print("dataloader for training created")