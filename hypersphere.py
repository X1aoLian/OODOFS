

from model import GraphModel
import torch
from torch import nn



class HyperSphere_Construction:
    def __init__(self,  reduced_size = 20, layer_num =3, Epoch =30, rec_weight = 1):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reduced_size = reduced_size
        self.layer_num = layer_num
        self.BCELoss = nn.BCELoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.Epoch = Epoch
        self.rec_weight = rec_weight
        self.predictionlist = []

    def First_Construction(self,graph_data, mask, feature_num, data):
        edge_index_size = graph_data.edge_index.size()
        self.model = GraphModel(layer_num=self.layer_num, in_features=self.reduced_size, hidden_features=64, edge_index_size=edge_index_size,
                                num_classes=2).to(self.device)

        self.mask = mask

        self.graph_data = graph_data.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.masklabel = graph_data.y.clone().detach()
        self.masklabel[~self.mask] = -1

        class_counts = torch.bincount(self.masklabel[self.mask].long())
        weights = len(self.masklabel[self.mask]) / (class_counts * len(class_counts)).to(self.device)
        self.CrossEntropyLoss = nn.CrossEntropyLoss(weight=weights)

        for _ in range(self.Epoch):
            W, X, Pred = self.model(graph_data.detach(), feature_num)
            self.center = torch.mean(X[mask][self.masklabel[self.mask] != 0], dim=0)
            self.radius = torch.max(X[mask][self.masklabel[self.mask] != 0])

            dist = torch.sqrt(torch.sum((X - self.center) ** 2, dim=1))
            #
            optimizer.zero_grad()
            inside_loss = torch.mean(torch.sum(torch.where((self.masklabel == 1) | (self.masklabel == -1),
                                                (torch.max(torch.tensor(0.0), dist - self.radius)),
                                                torch.tensor(0.0))))

            outside_loss = torch.mean(torch.sum(torch.where(self.masklabel == 0, (self.radius - dist) , torch.tensor(0.0))))
            dist_loss = inside_loss + outside_loss
            pred_loss = self.CrossEntropyLoss(Pred[self.mask], self.graph_data.y[self.mask].long())
            total_loss = 0.1 * dist_loss +  pred_loss
            total_loss.backward(retain_graph=True)
            optimizer.step()

            pred = torch.argmax(Pred, dim=1)
        self.predictionlist.extend(pred.detach().cpu().numpy())


    def Second_Construction(self,graph_data, mask, feature_num, old_feature_num):
        edge_index_size = graph_data.edge_index.size()
        self.model = GraphModel(layer_num=self.layer_num, in_features=self.reduced_size,
                                hidden_features=128, edge_index_size=edge_index_size,
                                num_classes=2).to(self.device)

        self.mask = mask
        self.graph_data = graph_data.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.masklabel = graph_data.y.clone().detach()
        self.masklabel[~self.mask] = -1
        class_counts = torch.bincount(self.masklabel[self.mask].long())
        weights = len(self.masklabel[self.mask]) / (class_counts * len(class_counts)).to(self.device)
        self.CrossEntropyLoss = nn.CrossEntropyLoss(weight=weights)
        W_old = graph_data.x[:feature_num]
        for _ in range(self.Epoch):
            self.testpredictionlist = []
            W, X, Pred = self.model(graph_data.detach(), feature_num)

            rec_loss = self.SmoothL1Loss(W.detach(), W_old.detach())

            W_old = W
            self.center = torch.mean(X[mask][self.masklabel[self.mask] != 0], dim=0)
            self.radius = torch.max(X[mask][self.masklabel[self.mask] != 0])
            dist = torch.sqrt(torch.sum((X - self.center) ** 2, dim=1))
            #
            optimizer.zero_grad()
            inside_loss = torch.mean(torch.sum(torch.where((self.masklabel == 1) | (self.masklabel == -1),
                                                (torch.max(torch.tensor(0.0), dist - self.radius)),
                                                torch.tensor(0.0))))
            outside_loss = torch.mean(torch.sum(torch.where(self.masklabel == 0, self.radius-dist, torch.tensor(0.0))))


            dist_loss = inside_loss + outside_loss
            pred_loss = self.CrossEntropyLoss(Pred[self.mask], self.graph_data.y[self.mask].long())

            total_loss = 0.1 * dist_loss + pred_loss + self.rec_weight * rec_loss
            total_loss.backward(retain_graph=True)
            optimizer.step()
            pred = torch.argmax(Pred, dim=1)

        self.predictionlist.extend(pred.detach().cpu().numpy())

    def load_model(self, model):
        state_dict_load = torch.load(self.path_state_dict)
        model.load_state_dict(state_dict_load)
        return model





