import os
import itertools
import torch
import json
import pandas as pd
import argparse
import multiprocessing 
from multiprocessing import set_start_method
from torch_geometric import seed_everything
from model.ada_med_graph import AdaGNN, ppmi_data_prepare

label_tar=['Label_NHY', 'Label_NP3_Tremor']
parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')

# parser.add_argument('--label', type=int, default=0, help='label')
parser.add_argument('--seed', type=int, default=42, help='label')
parser.add_argument('--month', type=str, default='12', help='time')
parser.add_argument('--dataset', type=str, default='ppmi', help='which dataset to use, ppmi or pdbp')
# parser.add_argument("--output_dir", type=str, default=os.getenv("AMLT_OUTPUT_DIR", "/tmp"))
parser.add_argument("--output_dir", type=str, default='new_results/')

args = parser.parse_args()
task =args.month
seed = args.seed

def tcsv(result):
    finale_r = result[-3]
    if type(finale_r) is dict:
        ind_c = ['micro auc', 'macro auc', 'weighted auc', 'macro f1-score', 'macro precision', 'macro recall', 
                'weighted f1-score', 'weighted precision', 'weighted recall', ]
        train_rel = []
        f_train_auc = finale_r['train_auccs']
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
        f_val_auc = finale_r['val_auccs']
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
        f_test_auc = finale_r['test_auccs']
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

    return result_df


def grid_search_trails(label_id):
    seed_everything(seed)
    # label_id =0
    if args.dataset =='ppmi':
        # data, label, features, edges, train_idx, val_idx, test_idx= ppmi_data_prepare(task = task, label_tar=label_tar, label_ind=label_id,fs_methods=fs_method, fs_hyper=fs_hyper,data_type=data_type )
        data, label, features, edges, train_idx, val_idx, test_idx= ppmi_data_prepare(task = task, label_tar=label_tar, label_ind=label_id)
        
        pdbp_idx = []
   
    hid_dim = [256, 512] 
    dropout =[0.1]
    k = [3,5]
    alpha = [0.1, 0.5]
    weight_decay = [10e-5, 10e-4]
    single_lrs = [0.0001, 0.00001]
    best_auc =0.6

    for paras in itertools.product(hid_dim, dropout,k,alpha,weight_decay, single_lrs):
        h = paras[0]
        d = paras[1]
        kk = paras[2]
        a = paras[3]
        wd = paras[4]
        slr = paras[5]

        ada_appnp = AdaGNN(data=data, label=label, features=features, 
                edge_features = edges, train_idx=train_idx, val_idx=val_idx, 
                test_idx=test_idx, external_idx=pdbp_idx,
                lr=0.3, hid_dim=h, dropout=d, k=kk, alpha=a, total_epoch=100, 
                single_lr=slr, weight_decay=wd, use_weight_loss=True, 
                n_estimators=10, early_stopping=20)
        tree_rec, model, appnp_logs = ada_appnp.TrainAdaboost()

        if len(model)>0:
            result = tree_rec[-1]
            val_auc_micro = result['val_auccs'][-1]
            # val_f1_micro = result['val_report']['weighted avg']['f1-score']

            if val_auc_micro> best_auc :
                best_auc = val_auc_micro
                tree_rec.append(['hid', 'dropout', 'k','alpha', 'weight decay', 'lr' ])
                tree_rec.append([h, d,k, a, wd, slr])
                result_df = tcsv(tree_rec)
                save_path = args.output_dir +'/'+label+'/'
                if os.path.isdir(save_path) == False:
                    os.makedirs(save_path)
                save_f = save_path+ task+'_wauc_'+str(val_auc_micro)[:5]+'seed_'+str(seed)+'h_'+str(h)+'d_'+str(d)+'al_'+str(a)+'.csv'
                save_1 = save_path+task+'_wauc_'+str(val_auc_micro)[:5]+'seed_'+str(seed)+'log_appnp'+'.json'
                save_fj = save_path+ task+'_wauc_'+str(val_auc_micro)[:5]+'seed_'+str(seed)+'h_'+str(h)+'d_'+str(d)+'al_'+str(a)+'lr_'+str(slr)+'.json'
               
                # save_2 = save_path+'wauc_'+str(val_auc_micro)[:5]+'seed_'+str(args.seed)+fs_method+str(fs_hyper)+'log_appnp_select'+'.json'
                f = open(save_1,"w")
                f.write(str(appnp_logs) )
                f.close()

                # f = open(save_2,"w")
                # f.write(str(appnp_select_logs) )
                # f.close()
                result_df.to_csv(save_f, index=False)
                f = open(save_fj, "w")
                f.write(str(tree_rec) )
                f.close()
    # save the mode
                # save the models
                model_f = save_path+ task+'_wauc_'+str(val_auc_micro)[:5]+'seed_'+str(seed)+'h_'+str(h)+'d_'+str(d)+'al_'+str(a)
                if os.path.isdir(model_f) == False:
                    os.makedirs(model_f)
                for i in range(len(model)):
                    torch.save(model[i], model_f+'/model_'+str(i)+'.pt')

# grid_search_trails(0)
if __name__ == '__main__':
    
    set_start_method('spawn')
    # labels = [42, 3407, 21]
    labels = [0,1]
    pro_list = []
    for l in labels:
        pro_list.append(l) 
    # creat prcesses
    processes = [multiprocessing.Process(target=grid_search_trails, args = (program, )) for program in pro_list]
    # start
    for p in processes:
        p.start()
    # join to wai

                            
                            
                        