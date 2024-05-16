import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch_scatter import scatter

import math


class GraphConvolution(Module):
    def __init__(self, in_feature, out_feature, bias = True):
        super(GraphConvolution, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        # init weight parameter
        self.weight = Parameter(torch.FloatTensor(in_feature, out_feature), requires_grad=True)
        # init bias parameter
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_feature))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # glorot initializaiotn 
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        # input img feature * weight matrix
        #support = torch.mm(input, self.weight)
        support = torch.einsum('bij, jk -> bik', input, self.weight) # handle high dim input
        # sparse matrix multiplication with adjacent matrix
        output = torch.bmm(adj, support)  
        # perform bias 
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        


class GCN(Module):
    def __init__(self, fea_dim, class_num, global_cluster_num, dropout):
        super(GCN, self).__init__()
        """
        fea_dim: feature dimension
        class_num: number of class for classification
        global_cluster_num: number of label of global cluster
        dropout: dropout value
        """
        # three graph convolution layers
        gc_out_dim = 16
        self.gc1 = GraphConvolution(fea_dim, fea_dim * 2)
        self.gc2 = GraphConvolution(fea_dim * 2, fea_dim * 4)
        self.gc3 = GraphConvolution(fea_dim * 4, gc_out_dim)

        self.dropout = dropout

        # classification layer
        hidden_dim = 512        
        self.classify_layer = nn.Sequential(
            nn.Linear(16 * global_cluster_num, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, class_num),
        )



    def forward(self, x, adj_matrix, mask_matrix, groups):
        x = F.relu(self.gc1(x, adj_matrix))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj_matrix))
        x = self.gc3(x, adj_matrix)

        # feature aggregation
        x = scatter(x, groups.long(), dim=1, reduce="max")

        # prepare the data from gcn for model of classification 
        batch, _, fea_dim = x.shape
        mask_matrix = (mask_matrix.unsqueeze(-1)).repeat(1,1,fea_dim)
        x = x * mask_matrix # update weight based on missing node infor

        x = x.view(batch, -1)

        # model for classification
        x = self.classify_layer(x)
        #x = torch.sigmoid(x)  #implemented in BCEWithLogitsLoss()

        return x


if __name__ == '__main__':
    pass







    