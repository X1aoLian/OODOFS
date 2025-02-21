
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from load_data import *
from graph_generation import graph_generation, graph_incremental
from hypersphere import HyperSphere_Construction
from metric import mask_list
from parameter import *





import numpy as np

import numpy as np
import os



def train(data_list, label_list,model_layers, training_epoch, reduced_size, label_rate, path):

    truth_label = []
    Starter = HyperSphere_Construction(reduced_size=reduced_size,layer_num=model_layers, Epoch=training_epoch)

    for times, data in enumerate(data_list):

        if times == 0:

            label = label_list[times]

            mask_data = mask_list(data, label_rate)
            sample_num, feature_num = data.size()

            graph_data = graph_generation(data, reduced_size, label)
            truth_label.extend(graph_data.y.numpy())
            Starter.First_Construction(graph_data = graph_data, mask = mask_data, feature_num = feature_num, data = data)

        else:

            label = label_list[times]
            mask_data = mask_list(data, label_rate)
            old_sample_num, old_feature_num = sample_num, feature_num
            sample_num, feature_num = data.size()
            graph_data = graph_incremental(graph_data.x[:old_feature_num].detach().cpu(), data, label, feature_num - old_feature_num, reduced_size)
            truth_label.extend(graph_data.y.numpy())
            Starter.Second_Construction(graph_data=graph_data, mask=mask_data, feature_num= feature_num, old_feature_num=old_feature_num)
    print('Acc ={}, F1 = {}, Recall={}, Precision={}'.format(
        accuracy_score(y_true=np.array(truth_label),
                       y_pred=np.array(Starter.predictionlist)),
        f1_score(y_true=np.array(truth_label),
                 y_pred=np.array(Starter.predictionlist), pos_label=0),
        recall_score(y_true=np.array(truth_label),
                     y_pred=np.array(Starter.predictionlist), pos_label=0),
        precision_score(y_true=np.array(truth_label),
                        y_pred=np.array(Starter.predictionlist),
                        pos_label=0)))





if __name__ == '__main__':


    random_seed = 1020

    f1_list = []
    precision = []
    recall = []
    data_path = "../data/3_backdoor.npz"
    data_list, label_list = tkde_newdataset_normalization(data_path,random_seed=1020)

    model_layers = 2
    training_epoch = 100
    reduced_size = return_reduced_size('backdoor')
    label_rate = 0.2
    train(data_list, label_list, model_layers, training_epoch, reduced_size, label_rate, path='backdoor')








