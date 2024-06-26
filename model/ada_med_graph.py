import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_classif

# from gpu_utils import gpu_init
from tqdm import tqdm
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
# from focal import MultiCEFocalLoss
from torch_geometric.data import Data
from torch_geometric.nn.conv import APPNP


#graph buliding functions
# define similarity of two patient
def Sim_func(a1,a2,thresh): 
    c_score = 0
    if abs(a1-a2) <= thresh:
        c_score +=1
    return c_score

def age_adj_matrix(patient_info, edge_f, thresh):
    edge_feature = patient_info[edge_f].to_list()
    edge_list=[]
    edge_wight=[]
    n_sample = len(edge_feature)
    adj = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(n_sample):
            adj[i,j] = Sim_func(edge_feature[i],edge_feature[j], thresh)
            if adj[i,j] != 0:
                edge_list.append([int(i),int(j)])
                edge_wight.append(adj[i,j])
    return adj, edge_list,edge_wight

# useful functions
def auc_multi(label, prob,  ave='micro'):

    return metrics.roc_auc_score(label, prob, multi_class='ovr', average=ave)


def F1_multi(label, pred,  ave='micro'):

    return metrics.f1_score(label, pred,  average=ave)

def cls_report_multi(label, pred):
    target_names = ['better','no change', 'worse']

    return metrics.classification_report(label, pred, output_dict =True, target_names=target_names)


def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))


def age_graph_bulider(all_data, label, feature_cols, edge_f, thresh):
    # save the labels
    norm_label_sh = all_data[label]
    labels_sh = torch.from_numpy(norm_label_sh.to_numpy()).long()
    node_feature_sh = torch.from_numpy(all_data[feature_cols].to_numpy()).float()
    adj_sh, edge_list_sh, edge_wight_sh = age_adj_matrix(all_data, edge_f, thresh)
    edge_list_sh = torch.tensor(edge_list_sh)
    # print(edge_list_sh)
    g_sh = Data(x = node_feature_sh, edge_index=edge_list_sh.t().contiguous(), y = labels_sh)
    return g_sh

class APPNP_model(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats=3, dropout=0, k=10, alpha=0.1,edge_drop=0, normalize =False):
        super(APPNP_model, self).__init__()
        self.lin1 = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_feats, hid_feats),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hid_feats, out_feats),
            )
        self.drop_out = nn.Dropout(p=dropout)
        self.l1 = APPNP(K=k, alpha=alpha, dropout=edge_drop)
        # self.active = nn.Softmax(dim=1)

    def forward(self, x, edge):
        h = self.lin1(x)
        # h = self.drop_out(h)
        logits = self.l1(h, edge)
        return logits


class AdaGNN:
    def __init__(self, data, label, features, edge_features, train_idx, val_idx, test_idx, external_idx=[],
                  lr=0.3, hid_dim=512, dropout=0.1, k=5, alpha=0.1, total_epoch=50, single_lr=0.1, 
                  weight_decay=0, use_weight_loss=False,  threshods = [0.5, 0.25, 0.125, 0.0625], n_estimators=10, early_stopping=10):
        # data: pandas dataframe with features and labels
        # label: name of label
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        # self.device = gpu_init(ml_library="torch")
        # self.model = APPNP_model(in_feats= len(features), hid_feats=hid_dim, dropout=dropout, k=k, alpha=alpha).to(self.device)
        self.data = data
        self.label = label
        self.threshods = threshods
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
        
        
    def train_single_appnp(self,  g, d_train, d_val, d_test, loss_weight):
        train_losses = []
        val_losses = []
        test_losses = []
        train_errors = []
        val_errors = []
        test_errors = []

        loggs = dict()
        model =  APPNP_model(in_feats= len(self.features), hid_feats=self.hid_dim, 
                             dropout=self.dropout, k=self.k, alpha=self.alpha).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(),lr=self.appnp_lr, weight_decay=self.weight_decay)    
        if self.weight_loss:
            criterion = torch.nn.CrossEntropyLoss(torch.FloatTensor(loss_weight).to(self.device))
        else:
            criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-8)

        features = g['x'].to(self.device)
        labels = g['y'].to(self.device)
        train_label = labels[self.train_idx].cpu().detach().numpy()
        val_label = labels[self.val_idx].cpu().detach().numpy()
        test_label = labels[self.test_idx].cpu().detach().numpy()

        edge = g['edge_index'].to(self.device)
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
            scheduler.step(loss_val)
            val_losses.append(loss_val.item())
            loss_test = criterion(output[self.test_idx], labels[self.test_idx])
            test_losses.append(loss_test.item())
            # scheduler.step(loss_val)
            if loss_l> loss_val:
                loss_l = loss_val
                best_epoch =epoch
            if self.early_stopping > 0 and(epoch - best_epoch)> self.early_stopping:
                break
            predict_in = torch.argmax(output, dim = -1).cpu().detach().numpy() # record the predict labels
            train_pred = predict_in[self.train_idx]
            trainincorrect = train_pred != train_label
            Train_Error = np.dot(trainincorrect, d_train) / np.sum(d_train, axis=0)
            val_pred = predict_in[self.val_idx]
            vincorrect = val_pred != val_label
            Val_Error = np.dot(vincorrect, d_val) / np.sum(d_val, axis=0)
            test_pred = predict_in[self.test_idx]
            testincorrect = test_pred != test_label
            test_Error = np.dot(testincorrect, d_test) / np.sum(d_test, axis=0)
            train_errors.append(Train_Error)
            val_errors.append(Val_Error)
            test_errors.append(test_Error)

            if min_loss > Val_Error:
                min_loss = Val_Error
                predict = output
                best_model_epoch=epoch
                Final_train_error = Train_Error
                Final_val_error = Val_Error
                Final_test_error = test_Error

        prob = predict.cpu().detach().numpy()
        predict = torch.argmax(predict, dim = -1) # record the predict labels
        predict = predict.cpu().detach().numpy()
    
        loggs['training loss'] = train_losses
        loggs['validation loss'] = val_losses
        loggs['test loss'] = test_losses
        loggs['Train error'] = train_errors
        loggs['Val error'] = val_errors
        loggs['Test error'] = test_errors
        loggs['model epoch'] = best_model_epoch
        return predict, prob, Final_train_error, Final_val_error, Final_test_error, model, loggs
 
    
    def best_appnp(self, D_train, D_val, D_test, loss_weight,  remove_edge =[]):
        # appnp_sellog = dict()
        edge_f_list = list(set(self.edge_features)-set(remove_edge))
        losss=100 
        for edge_f in edge_f_list:
            for thr in self.threshods:
                # if self.thresh_all:
                thresh = self.data[edge_f].quantile(thr)
                g = age_graph_bulider(self.data, self.label, self.features, edge_f, thresh)
                predict, prob, Train_Error, Val_Error, Test_Error, model, logf= self.train_single_appnp(g, D_train, D_val, D_test,loss_weight)                                                                                    
                # appnp_sellog[edge_f+'_'+str(thr)] =[Train_Error, Val_Error]
                
                if Train_Error < losss:
                    losss = Train_Error
                    best_pred, best_pob, min_train_error, min_val_error, min_test_error, best_model, appnplog = predict, prob, Train_Error,Val_Error, Test_Error, model, logf
                    ed_f = edge_f
                    thd = thr
        return best_pred, best_pob, min_train_error, min_val_error, min_test_error, best_model, appnplog, ed_f, thd
 

    def TrainAdaboost(self):
        idx_train = self.train_idx
        idx_val = self.val_idx
        idx_test = self.test_idx
        idx_pdbp = self.pdbp_idx
        weak_models =[]
        min_error_rate = 1-1/3.0 # SAMME
        alpha_sum = 0
        best_auc =0
        LearnerHistory = [] #保存基学习器的列表
        n_samples = len(self.data)
        Predict = np.zeros((n_samples,3))
        Probe = np.zeros((n_samples,3))
        #开始集成学习的训练过程
        y_clss = np.array(self.data[self.label])
        clss = np.array(sorted(list(set(y_clss))))
        classes = clss[:, np.newaxis]
        y_train = np.array(y_clss[idx_train])
        y_val = np.array(y_clss[idx_val])
        y_test = np.array(y_clss[idx_test])
        y_pdbp = np.array(y_clss[idx_pdbp])
        loss_weight=class_weight.compute_class_weight('balanced',classes =np.unique(y_clss),y=y_clss)
        loss_weight1 = class_weight.compute_sample_weight(class_weight='balanced',y=y_clss)
        D_train = np.ones((len(idx_train)))*(1/len(idx_train)) *loss_weight1[idx_train]
        D_val = np.ones((len(idx_val)))*(1/len(idx_val)) *loss_weight1[idx_val]
        D_test = np.ones((len(idx_test)))*(1/len(idx_test)) *loss_weight1[idx_test]

        remove_e = []
        appnp_logs = []
        # appnp_select_logs = []
        for i in tqdm(range(self.n_estimators)):
            PredictLabel, PredictProb, Train_Error, Val_Error, Test_Error, model, appnp_log, edge_f, thresh = self.best_appnp(D_train, D_val, D_test,loss_weight,remove_e)        
            appnp_logs.append(appnp_log) # record the best appnp logs
            # appnp_select_logs.append(appnp_sellog)
            #当期望误差大于thresh 时候，退出循环
            if (Train_Error> min_error_rate) or (Val_Error > min_error_rate):
                break
            remove_e.append(edge_f)
            #calculate the estimator weights
            alpha_train = self.lr*np.log((1-Train_Error)/Train_Error) + np.log(3-1)
            alpha_val = self.lr*np.log((1-Val_Error)/Val_Error) + np.log(3-1)
            alpha_test = self.lr*np.log((1-Test_Error)/Test_Error) + np.log(3-1)
            trainPredictLabel = PredictLabel[idx_train]
            valPredictLabel = PredictLabel[idx_val]
            testPredictLabel = PredictLabel[idx_test]

            train_incorrect = trainPredictLabel != y_train
            val_incorrect = valPredictLabel != y_val
            test_incorrect = testPredictLabel != y_test

            D_train *= np.exp(alpha_train * train_incorrect)
            D_val *= np.exp(alpha_val * val_incorrect)
            D_test *= np.exp(alpha_test * test_incorrect)

            sample_weight_sum_train = np.sum(D_train, axis=0)
            sample_weight_sum_val = np.sum(D_val, axis=0)
            sample_weight_sum_test = np.sum(D_test, axis=0)

            D_train /= sample_weight_sum_train
            D_val /= sample_weight_sum_val
            D_test /= sample_weight_sum_test

            #计算前epoch+1个基学习器的集成算法的误差
            alpha_sum += alpha_train
            # calcalate the prediction label
            Predict = Predict+(PredictLabel==classes).T * alpha_train # here may it into 3-d encoding!!
            Predict /= alpha_sum
            Predict_final = np.argmax(Predict, axis=1)
            trainPre = Predict_final[idx_train] # may need to change here, sign function may not be a good idea
            valPre = Predict_final[idx_val]
            testPre = Predict_final[idx_test]
            PDBPPre = Predict_final[idx_pdbp]
            
            # calcalate the prediction probs
            Probe = Probe+(PredictProb) * alpha_train # here may it into 3-d encoding!!
            Probe /= alpha_sum
            Probe = np.exp((1. / (3 - 1)) * Probe)
            normalizer = Probe.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            Probe /= normalizer
            trainPro = Probe[idx_train]
            valPro = Probe[idx_val]
            testPro = Probe[idx_test]
            PDBPPro = Probe[idx_pdbp]
            train_auc_micro = auc_multi(y_train, trainPro, 'micro')
            val_auc_micro = auc_multi(y_val, valPro, 'micro')
            test_auc_micro = auc_multi(y_test, testPro, 'micro')
            train_auc_macro = auc_multi(y_train, trainPro, 'macro')
            val_auc_macro = auc_multi(y_val, valPro, 'macro')
            test_auc_macro = auc_multi(y_test, testPro, 'macro')
            train_auc_weight = auc_multi(y_train, trainPro, 'weighted')
            val_auc_weight = auc_multi(y_val, valPro, 'weighted')
            test_auc_weight = auc_multi(y_test, testPro, 'weighted')
            train_report = cls_report_multi(y_train, trainPre)
            val_report = cls_report_multi(y_val, valPre)
            test_report = cls_report_multi(y_test, testPre)
            train_f1_micro = F1_multi(y_train, trainPre, 'micro')
            val_f1_micro = F1_multi(y_val, valPre, 'micro')
            test_f1_micro = F1_multi(y_test, testPre, 'micro')

            train_auccs = [train_auc_micro, train_auc_macro, train_auc_weight]
            val_auccs = [val_auc_micro, val_auc_macro, val_auc_weight]
            test_auccs = [test_auc_micro, test_auc_macro, test_auc_weight]
            train_errorate = get_error_rate(trainPre, y_train)
            val_errorate = get_error_rate(valPre, y_val)
            test_errorate = get_error_rate(testPre, y_test)

            BestGraph = {}
            BestGraph['edge'] = edge_f
            BestGraph['thred'] = thresh
            BestGraph['alpha'] = alpha_train
            BestGraph['Error rate'] = [train_errorate, val_errorate, test_errorate]
            BestGraph['train_auccs'] = train_auccs
            BestGraph['train_report'] = train_report
            BestGraph['train_f1_micro'] = train_f1_micro
            BestGraph['val_auccs'] = val_auccs
            BestGraph['val_report'] = val_report
            BestGraph['val_f1_micro'] = val_f1_micro
            BestGraph['test_auccs'] = test_auccs
            BestGraph['test_report'] = test_report
            BestGraph['test_f1_micro'] = test_f1_micro
            if len(idx_pdbp)>0:
                pdbp_f1_micro = F1_multi(y_pdbp, PDBPPre, 'micro')
                pdbp_report = cls_report_multi(y_pdbp, PDBPPre)
                pdbp_auc_micro = auc_multi(y_pdbp, PDBPPro, 'micro')
                pdbp_auc_macro = auc_multi(y_pdbp, PDBPPro, 'macro')
                pdbp_auc_weighted = auc_multi(y_pdbp, PDBPPro, 'weighted')
                pdbp_auccs = [pdbp_auc_micro, pdbp_auc_macro, pdbp_auc_weighted]
                BestGraph['pdbp_auccs'] = pdbp_auccs
                BestGraph['pdbp_report'] = pdbp_report
                BestGraph['PDBP Micro F1'] = pdbp_f1_micro
            if val_auccs[-1] > best_auc:
                best_auc = val_auccs[-1]
                LearnerHistory.append(BestGraph)
                weak_models.append(model.state_dict()) 
            else:
                break
        if len(weak_models)>0:
            return LearnerHistory, weak_models, appnp_logs
        else:
            return [], [], []

def scale_data(X):
    preproc = MinMaxScaler()
    cols = X.columns
    return pd.DataFrame(preproc.fit_transform(X), columns=cols)

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
    features = list(np.loadtxt(path+'ppmi_node.txt', dtype= str))
    # edges = list(np.loadtxt(path+'ppmi_edge.txt', dtype= str))
    edges = features

    return pd_patients_all, c_label, features, edges, train_idx, val_idx, test_idx




def pdbp_data_prepare(task = '24month', data_type ='strict',label_tar=[], label_ind=0, fs_methods='PCA', fs_hyper=0.8):
    # read data
    c_label = label_tar[label_ind]
    path = 'data/'
    train_all = pd.read_csv(path+'ppmi_'+task+'_train_'+data_type+'_pdbp.csv')
    train_all_idx = list(train_all.index)
    test_data = pd.read_csv(path+'ppmi_'+task+'_test_'+data_type+'_pdbp.csv')
    train_all = train_all.sample(frac=1.0).reset_index(drop=True) # shaffle your dataset!!!!
    f = [train_all, test_data]
    pd_patients_all = pd.concat(f)
    pd_patients_all.reset_index(inplace=True, drop=True)
    pd_patients_all['ENROLL_AGE'] = (pd_patients_all['ENROLL_AGE']-pd_patients_all['ENROLL_AGE'].min())/(pd_patients_all['ENROLL_AGE'].max()-pd_patients_all['ENROLL_AGE'].min())
    # read PBDP data
   
    train_data = pd_patients_all.loc[train_all_idx]
    train_idx = list(train_data.sample(frac=0.8).index)
    val_data= train_data.drop(train_idx)
    val_idx= list(val_data.index)
    test_idx = list(set(pd_patients_all.index)- set(train_all_idx))
    pdbp_data = pd.read_csv(path+'pdbp_'+task+'_'+data_type+'.csv')
    pdbp_data.drop(columns=['EVENT_ID', 'Change_NHY',  'INFODT', 'Change_MoCA_score', 'Change_NPTOT', 'Change_NP2PTOT', 'Change_NP3_Axial', 'Change_NP3_Tremor', 'UPSIT_TOT', 'Change_NP3_Akinetic_Rigid', 'Change_NP1TOT', 'Change_ESS_TOT', 'Change_NP3TOT','PRS83'], inplace=True)
    feature_cols = set(pdbp_data.columns)-set(label_tar)-set(['PATNO']) # including sex and age
    feature_cols =list(feature_cols)
    pdbp_data[feature_cols] =scale_data(pdbp_data[feature_cols])
    pd_patients_all = pd_patients_all[pdbp_data.columns]
    f = [pd_patients_all, pdbp_data]
    pd_patients_all_new = pd.concat(f)
    pd_patients_all_new.reset_index(inplace=True)
    pdbp_idx = list(set(pd_patients_all_new.index)- set(train_all_idx)- set(test_idx))
    
    if fs_methods == 'lda':
        pd_label = pd_patients_all_new[label_tar]
        lda =LDA(n_components=fs_hyper)
        train_data = lda.fit_transform(train_data[feature_cols], train_data[c_label])
        new_feature_cols = ['lda_'+str(i) for i in range(fs_hyper)]
        train_data = pd.DataFrame(train_data, columns=new_feature_cols)
        train_data=scale_data(train_data)
        test_data = lda.transform(test_data[feature_cols])
        test_data = pd.DataFrame(test_data, columns=new_feature_cols)
        test_data=scale_data(test_data)
        pdbp_data = lda.transform(pdbp_data[feature_cols])
        pdbp_data = pd.DataFrame(pdbp_data, columns=new_feature_cols)
        pdbp_data=scale_data(pdbp_data)

        f = [train_data, test_data, pdbp_data]
        pd_patients_all_new = pd.concat(f)
        pd_patients_all_new.reset_index(inplace=True, drop=True)
        pd_patients_all_new[label_tar] = pd_label
        features = new_feature_cols
        edges= new_feature_cols
    elif fs_methods == 'anova':
        pd_label = pd_patients_all_new[label_tar]
        select = SelectKBest(f_classif,  k=int(fs_hyper))
        select.fit(train_data[feature_cols], train_data[c_label])
        new_feature_cols = train_data[feature_cols].columns[select.get_support()]

        train_data = train_data[new_feature_cols]
        test_data = test_data[new_feature_cols]
        pdbp_data = pdbp_data[new_feature_cols]
        f = [train_data, test_data, pdbp_data]
        pd_patients_all_new = pd.concat(f)
        pd_patients_all_new.reset_index(inplace=True, drop=True)
        pd_patients_all_new[label_tar] = pd_label
        features = new_feature_cols
        edges= new_feature_cols
    else:
    # load important features
        features = feature_cols
        edges= feature_cols
    return pd_patients_all_new, c_label, features, edges, train_idx, val_idx, test_idx, pdbp_idx
