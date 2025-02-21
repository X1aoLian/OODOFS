import torch
import torch.utils.data as Data
from torch_geometric.data import Data
from original.datasets import *


def create_data(data, embedding_matrix):
    W = embedding_matrix
    rec_samples = []
    sum = 0
    for element in data:
        for i, feature in enumerate(element):
            sum += feature * W[i]
        rec_samples.append(sum)
        sum = 0
    return rec_samples

def create_node(data,  reduced_size):
    sample_num, feature_num = data.shape
    W = np.random.rand(feature_num, reduced_size)
    reconstructed_data = np.zeros((sample_num, reduced_size))  # 初始化重构后的数据集
    for i in range(sample_num):
        # 对于每个样本，计算它与嵌入矩阵中对应向量的乘积之和
        for j in range(feature_num):
            reconstructed_data[i] += data[i, j].numpy() * W[j]
    W = torch.Tensor(W)
    reconstructed_data = torch.Tensor(reconstructed_data)
    node_feature = torch.cat((W, reconstructed_data), dim=0)
    return node_feature

def create_edge(x, feature_num):
    init_source, init_end = torch.nonzero(x).transpose(0, 1)

    edge_source = init_source + feature_num
    edge_end = init_end
    edge_source_ = torch.cat([edge_source, edge_end], dim=0)

    edge_end_ = torch.cat([edge_end, edge_source], dim=0)
    edge_index = torch.stack([edge_source_, edge_end_], dim=0) # [2, edge_num]
    return edge_index

def create_label(label):
    y = []
    for i in label:
        if i == 1:
            y.append(1)
        else:
            y.append(0)
    y = torch.Tensor(y)
    return y

def graph_generation(x, reduced_size, y):
    sample_num, feature_num = x.size()
    node_feature = create_node(x, reduced_size)

    edge_index = create_edge(x, feature_num)
    y = create_label(y)
    data = Data(x=node_feature, edge_index=edge_index, y = y)
    return data

def graph_incremental(old_embedding_vectors, new_samples, new_label, new_feature_num, reduced_size):
    sample_num, feature_num = new_samples.shape
    new_graph_edge = create_edge(new_samples, feature_num)
    new_label = create_label(new_label)


    if new_feature_num != 0:
        new_embedding_vectors = torch.rand(new_feature_num, reduced_size)
        embedding_matrix = torch.cat((old_embedding_vectors, new_embedding_vectors), dim = 0)
    else:
        embedding_matrix = old_embedding_vectors.cpu()

    reconstructed_data = torch.zeros((sample_num, reduced_size))  # 初始化重构后的数据集
    for i in range(sample_num):
        # 对于每个样本，计算它与嵌入矩阵中对应向量的乘积之和
        for j in range(feature_num):
            reconstructed_data[i] += new_samples[i, j] * embedding_matrix[j]
    reconstructed_data = torch.Tensor(reconstructed_data)


    node_feature = torch.cat((embedding_matrix, reconstructed_data), dim = 0)



    data = Data(x = node_feature, edge_index = new_graph_edge, y = new_label)
    return data
