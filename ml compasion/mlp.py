import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt
import json
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report ,roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import argparse
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser(description=' mlp classification')
parser.add_argument('--task', type=str, default='12', help='label')
parser.add_argument('--selection', type=bool, default=False, help='label')
parser.add_argument('--n_class', type=int, default=3, help='label')

args = parser.parse_args()
task = args.task
n_class = args.n_class
sel = args.selection
if n_class == 3:
    data_type = '_new' # '_strict', '_full_imputed'

path = 'new_data/'
label_tar=['Label_NHY','Label_NP1TOT','Label_NP2PTOT','Label_NP3TOT', 'Label_NPTOT','Label_NP3_Axial','Label_NP3_Tremor',
'Label_NP3_Akinetic_Rigid','Label_MoCA_score','Label_ESS_TOT']
# label_tar=['Label_MoCA_score']

train_data = pd.read_csv(path+'ppmi_'+task+'_train''.csv')
test_data = pd.read_csv(path+'ppmi_'+task+'_test'+'.csv')
feature_cols = np.loadtxt(path+'ppmi_edge.txt', dtype=str)


x_train_all = train_data[feature_cols]
x_test = test_data[feature_cols]
y_train_all = train_data[label_tar]
train_idx = list(x_train_all.sample(frac=0.8).index)
x_train= x_train_all.loc[train_idx]
y_train_al= y_train_all.loc[train_idx]
x_val= x_train_all.drop(train_idx)
y_val_all = y_train_all.drop(train_idx)
y_test_all = test_data[label_tar]

def multi_class_metric(y_train, y_train_pred, y_train_pl,y_val, y_val_pred, y_val_pl, y_test, y_test_pred, y_test_pl):
        train_auc_micro = roc_auc_score(y_train, y_train_pred, multi_class='ovr', average='micro') # micro auc
        val_auc_micro = roc_auc_score(y_val, y_val_pred, multi_class='ovr', average='micro') # micro auc
        test_auc_micro = roc_auc_score(y_test, y_test_pred, multi_class='ovr', average='micro') # micro auc
        train_auc_macro = roc_auc_score(y_train, y_train_pred, multi_class='ovr', average='macro') # macro auc
        val_auc_macro = roc_auc_score(y_val, y_val_pred, multi_class='ovr', average='macro') # macro auc
        test_auc_macro = roc_auc_score(y_test, y_test_pred, multi_class='ovr', average='macro') # macro auc
        train_auc_weight = roc_auc_score(y_train, y_train_pred, multi_class='ovr', average='weighted') # macro auc
        val_auc_weight = roc_auc_score(y_val, y_val_pred, multi_class='ovr', average='weighted') # macro auc
        test_auc_weight = roc_auc_score(y_test, y_test_pred, multi_class='ovr', average='weighted') # macro auc
        train_auccs = [train_auc_micro, train_auc_macro, train_auc_weight] 
        val_auccs = [val_auc_micro, val_auc_macro, val_auc_weight] 
        test_auccs = [test_auc_micro, test_auc_macro, test_auc_weight] 
        target_names = ['better','no change', 'worse']
        train_report = classification_report(y_train, y_train_pl, output_dict =True, target_names=target_names)
        val_report = classification_report(y_val, y_val_pl, output_dict =True, target_names=target_names)
        test_report = classification_report(y_test, y_test_pl, output_dict =True, target_names=target_names)
        return train_auccs, val_auccs, test_auccs, train_report, val_report, test_report

def t_csv(finale_r):
        ind_c = ['micro auc', 'macro auc', 'weighted auc', 'macro f1-score', 'macro precision', 'macro recall', 
                'weighted f1-score', 'weighted precision', 'weighted recall', ]
        train_rel = []
        f_train_auc = finale_r['train_auccs: micro, macro, weighted']
        # print(f_train_auc)
        f_train_macro = finale_r['train_report']['macro avg']
        f_train_weigted = finale_r['train_report']['weighted avg']
        train_rel.append(f_train_auc[0])
        train_rel.append(f_train_auc[1])
        train_rel.append(f_train_auc[2])
        train_rel.append(f_train_macro['f1-score'])
        train_rel.append(f_train_macro['precision'])
        train_rel.append(f_train_macro['recall'])
        train_rel.append(f_train_weigted['f1-score'])
        train_rel.append(f_train_weigted['precision'])
        train_rel.append(f_train_weigted['recall'])

        val_rel = []
        f_val_auc = finale_r['val_auccs: micro, macro, weighted']
        f_val_macro = finale_r['val_report']['macro avg']
        f_val_weigted = finale_r['val_report']['weighted avg']
        val_rel.append(f_val_auc[0])
        val_rel.append(f_val_auc[1])
        val_rel.append(f_val_auc[2])
        val_rel.append(f_val_macro['f1-score'])
        val_rel.append(f_val_macro['precision'])
        val_rel.append(f_val_macro['recall'])
        val_rel.append(f_val_weigted['f1-score'])
        val_rel.append(f_val_weigted['precision'])
        val_rel.append(f_val_weigted['recall'])
        test_rel = []
        f_test_auc = finale_r['test_auccs: micro, macro, weighted']
        f_test_macro = finale_r['test_report']['macro avg']
        f_test_weigted = finale_r['test_report']['weighted avg']
        test_rel.append(f_test_auc[0])
        test_rel.append(f_test_auc[1])
        test_rel.append(f_test_auc[2])
        test_rel.append(f_test_macro['f1-score'])
        test_rel.append(f_test_macro['precision'])
        test_rel.append(f_test_macro['recall'])
        test_rel.append(f_test_weigted['f1-score'])
        test_rel.append(f_test_weigted['precision'])
        test_rel.append(f_test_weigted['recall'])
        result_df = pd.DataFrame()
        result_df['metric'] =ind_c
        result_df['train'] =train_rel
        result_df['val'] =val_rel
        result_df['test'] =test_rel
        return  result_df


def mlp(i,x_train_all= x_train_all, x_train= x_train, x_val=x_val, x_test= x_test):
        c_label = label_tar[i]
        # if sel:
        #       feature_cols = np.load('new_results/statistics/features/'+task+'_'+c_label+'.npy',  allow_pickle=True)
        #       x_train_all = x_train_all[feature_cols]
        #       x_train = x_train[feature_cols]
        #       x_val = x_val[feature_cols]
        #       x_test = x_test[feature_cols]            
        # print(c_label)
        y_train_ll = y_train_all[c_label]
        # print(y_train_ll)
        y_train =y_train_al[c_label]
        y_val =y_val_all[c_label]
        y_test =y_test_all[c_label]
        model = MLPClassifier(validation_fraction=0.2, early_stopping=True)
        pipeline = Pipeline([('model', model)])
        # define grid search
        grid = {
                'model__hidden_layer_sizes': [(16, 32,63), (32, 32), 
                                              (64, 32, 16), (128, 64, 16), 
                                              (256, 128, 128), (512, 256, 128)],
                'model__learning_rate_init':[0.3, 0.1, 0.05],
                'model__alpha':[0, 0.001, 0.0001, 0.00001],
                'model__random_state':[42, 77, 21]
                }
        # define evaluation procedure
        cvv = StratifiedKFold(5, shuffle=True)
        if n_class == 3:
                cv = GridSearchCV(pipeline, grid, scoring='roc_auc_ovr_weighted', cv=cvv, n_jobs=8, verbose=1)
        else:
                cv = GridSearchCV(pipeline, grid, scoring='roc_auc', cv=cvv, n_jobs=8, verbose=1)
              
        cv.fit(x_train_all, y_train_ll)
        print("Best params:", cv.best_params_)
        print("Best score:", cv.best_score_)
        # evaluate the pipeline
        y_train_pred = cv.predict_proba(x_train)
        y_val_pred = cv.predict_proba(x_val)
        y_test_pred = cv.predict_proba(x_test)
        y_train_pl = cv.predict(x_train)
        y_val_pl = cv.predict(x_val)
        y_test_pl = cv.predict(x_test)
          # report classi
        # print(y_train_pred)
        if n_class==3:
                train_auccs, val_auccs, test_auccs, train_report, val_report, test_report = multi_class_metric(y_train, y_train_pred, y_train_pl,y_val, y_val_pred, y_val_pl, y_test, y_test_pred, y_test_pl)
                return cv.best_params_, train_auccs, train_report,val_auccs, val_report, test_auccs, test_report

        else:
                train_auccs, val_auccs, test_auccs, train_report, val_report, test_report, train_pracu, val_prauc, test_prauc = binary_class_metric(y_train, y_train_pred, y_train_pl,y_val, y_val_pred, y_val_pl, y_test, y_test_pred, y_test_pl)
                return cv.best_params_, train_auccs, train_report, train_pracu, val_auccs, val_report, val_prauc, test_auccs, test_report, test_prauc

              


for i in range(len(label_tar)):

    if n_class ==3:
        file_name = 'new_results/mlp/ppmi/'+task+'/'+label_tar[i]
        best_param, train_auccs, train_report, val_auccs, val_report,test_auccs, test_report  = mlp(i)
        results = dict()
        results['best params'] = best_param
        results['train_auccs: micro, macro, weighted'] = train_auccs
        results['train_report'] = train_report

        results['val_auccs: micro, macro, weighted'] = val_auccs
        results['val_report'] = val_report
        results['test_auccs: micro, macro, weighted'] = test_auccs
        results['test_report'] = test_report
        result_df = t_csv(results)
        result_df.to_csv(file_name+'.csv', index =False)