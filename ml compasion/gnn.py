import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from itertools import product

from tqdm import tqdm
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric import seed_everything
from torch_geometric.nn.conv import APPNP
import itertools
from ada_s import ppmi_data_prepare, age_graph_bulider, APPNP_model



label_tar=['Label_NHY','Label_NP1TOT','Label_NP2PTOT','Label_NP3TOT','Label_NPTOT','Label_NP3_Axial','Label_NP3_Tremor',
        'Label_NP3_Akinetic_Rigid','Label_MoCA_score','Label_ESS_TOT']

def ppmi_data_prepare(task = '24month',  label_tar=[], label_ind=0):
    # read data
    path = 'new_data/'
    c_label = label_tar[label_ind]
    path = path
    # train_all = pd.read_csv(path+'ppmi_'+task+'_train'+'_strict.csv')
    train_all = pd.read_csv(path+'ppmi_'+task+'_train'+'.csv')
    train_all_idx = list(train_all.index)
    test_data = pd.read_csv(path+'ppmi_'+task+'_test'+'.csv')
    train_all = train_all.sample(frac=1.0).reset_index(drop=True) # shaffle your dataset!!!!
    f = [train_all, test_data]
    pd_patients_all = pd.concat(f)
    pd_patients_all.reset_index(inplace=True, drop=True)
    train_data = pd_patients_all.loc[train_all_idx]
    train_idx = list(train_data.sample(frac=0.8).index)
    val_data= train_data.drop(train_idx)
    val_idx= list(val_data.index)
    test_idx = list(set(pd_patients_all.index)- set(train_all_idx))
    features = list(np.loadtxt(path+'ppmi_edge.txt', dtype= str))
    edges = features

    return pd_patients_all, c_label, features, edges, train_idx, val_idx, test_idx


class GNN:
    def __init__(self, data, label, features, edge_features, train_idx, val_idx, test_idx, external_idx=[],
                  lr=0.3, hid_dim=512, dropout=0.1, k=5, alpha=0.1, total_epoch=50, single_lr=0.0001, 
                  weight_decay=0, use_weight_loss=False,  n_estimators=10, early_stopping=10):
        # label: name of label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = data
        self.label = label
        self.features= features
        self.edge_features = edge_features
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.pdbp_idx = external_idx
        self.lr=lr
        self.hid_dim= hid_dim
        self.dropout= dropout
        self.k=k
        self.alpha= alpha
        self.appnp_epochs = total_epoch
        self.appnp_lr =single_lr
        self.weight_decay =weight_decay
        self.weight_loss = use_weight_loss
        # self.use_focal_loss = use_focal_loss
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        
        
    def train_single_appnp(self, model, g, loss_weight=None,  weight_decay=0):
        train_losses = []
        val_losses = []
        test_losses = []
        features = g['x'].to(self.device)
        labels = g['y'].to(self.device)
        loggs = dict()
        optimizer = torch.optim.Adam(model.parameters(),lr=self.appnp_lr, weight_decay=weight_decay)    
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-9)
        # scheduler =torch.optim.lr_scheduler.StepLR(optimizer,10,gamma=0.1,last_epoch=-1,verbose=False)
        if self.weight_loss:
            criterion = torch.nn.CrossEntropyLoss(torch.FloatTensor(loss_weight).to(self.device))
        else:
            criterion = torch.nn.CrossEntropyLoss()
        edge = g['edge_index'].to(self.device)
        # print(edge)
        loss_l =100
        min_loss = 10
        best_epoch = 0
        best_model_epoch = 0

        for epoch in range(self.appnp_epochs):
            model.train()
            optimizer.zero_grad()
            output= model(features, edge)
            
            loss_train = criterion(output[self.train_idx], labels[self.train_idx])
            train_losses.append(loss_train.item())
            loss_train.backward()
            optimizer.step()
            loss_val = criterion(output[self.val_idx], labels[self.val_idx])
            # scheduler.step(loss_val)
            val_losses.append(loss_val.item())
            loss_test = criterion(output[self.test_idx], labels[self.test_idx])
            test_losses.append(loss_test.item())
            # scheduler.step(loss_val)
            if loss_l> loss_val:
                loss_l = loss_val
                best_epoch =epoch
            if self.early_stopping > 0 and(epoch - best_epoch)> self.early_stopping:
                break
            if min_loss > loss_val:
                min_loss = loss_val
                predict = output
                best_model_epoch = loss_val.item()
        prob = F.softmax(predict, dim=1)
        prob = prob.cpu().detach().numpy()
        predict = torch.argmax(predict, dim = -1) # record the predict labels
        predict = predict.cpu().detach().numpy()
        loggs['training loss'] = train_losses
        loggs['validation loss'] = val_losses
        loggs['test loss'] = test_losses
        loggs['model epoch'] = best_model_epoch
        labels = labels.cpu().detach().numpy()

        trainauc = metrics.roc_auc_score(labels[self.train_idx], prob[self.train_idx], multi_class='ovr', average='weighted')
        valauc = metrics.roc_auc_score(labels[self.val_idx], prob[self.val_idx], multi_class='ovr', average='weighted')
        testauc = metrics.roc_auc_score(labels[self.test_idx], prob[self.test_idx], multi_class='ovr', average='weighted')
        print(trainauc, valauc, testauc)
        return loggs, trainauc, valauc, testauc

def tune(label_ind, task ='12'):
    data, label, features, edges, train_idx, val_idx, test_idx = ppmi_data_prepare(task = task ,label_tar=label_tar, label_ind=label_ind)
    # data['Age'] = data['ENROLL_AGE']*(84.9-32.2) +32.2

    seed_ = [21, 42, 3407]
    thred = [5, 4.5, 4, 3, 2.5 ]
    hd = [256, 512]
    tp = [100]
    kk = [1,3, 5]
    early_stopping = [10, 15, 20]
    single_lr = [10e-5, 10e-5, 5*10e-4]
    best_val_auc = 0
    for s, t, h, tpp, k, e, slr in product(seed_, thred, hd, tp, kk, early_stopping, single_lr):
        seed_everything(s)
        ada_appnp = GNN(data=data, label=label, features=features, 
                        edge_features = edges, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                        lr=0.3, hid_dim=h, dropout=0.1, k=k, alpha=0.3, total_epoch=tpp, single_lr=slr, 
                        weight_decay=slr, use_weight_loss=True, n_estimators=10, early_stopping=e)
        model = APPNP_model(in_feats= len(features), hid_feats=ada_appnp.hid_dim, dropout=ada_appnp.dropout, 
                            k=ada_appnp.k, alpha=ada_appnp.alpha).cuda()
        y_clss = np.array(ada_appnp.data[ada_appnp.label])
        loss_weight=class_weight.compute_class_weight('balanced',classes =np.unique(y_clss),y=y_clss)
        g= age_graph_bulider(ada_appnp.data, ada_appnp.label, ada_appnp.features, 'Age', t)
        _, tra, val, tes = ada_appnp.train_single_appnp(model, g, loss_weight)
        if val > best_val_auc:
            best_val_auc = val
            best_age_v = t
            final_val_score = (tra, val, tes)
    return final_val_score, best_age_v, best_age_t

label_ids = [0,1,2,3,4,5,6,7,8,9]
task = '12'
train_scores = []
val_scores = []
test_scores = []
train2_scores = []
val2_scores = []
test2_scores = []
for i in label_ids:
    final_val_score ,best_age_v, best_age_t = tune(i, task = task)
    train_scores.append(final_val_score[0])
    val_scores.append(final_val_score[1])
    test_scores.append(final_val_score[2])

result_pd = pd.DataFrame()
result_pd['Labels'] = label_tar
result_pd['Train_auc'] = train_scores
result_pd['Val_auc'] = val_scores
result_pd['Test_auc'] = test_scores

result_pd.to_csv('new_results/gnn_ppmi_'+task+str(best_age_v)+'_'+str(best_age_t)+'month.csv', index=False)



