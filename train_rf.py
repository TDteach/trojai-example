import utils
import os
import pickle
import numpy as np

def read_repr_feature(folder, suffix_name):
    fns=os.listdir(folder)
    fns.sort()
    rf_fts=dict()
    for fn in fns:
        md_name=fn.split('_')[0]
        dummy_name=md_name+'_'+suffix_name+'.pkl'
        if fn!=dummy_name: continue

        pickle_path=os.path.join(folder,fn)
        with open(pickle_path,'rb') as f:
            data_dict=pickle.load(f)
        avg_repr_grads=data_dict['avg_repr_grads']
        #avg_embs_grads=data_dict['avg_embs_grads']
        #ft=np.concatenate([avg_embs_grads,avg_repr_grads],axis=0)
        ft=avg_repr_grads
        #ft=avg_embs_grads

        rf_fts[md_name]=ft
    return rf_fts


def read_grads_feature(folder, suffix_name):
    fns=os.listdir(folder)
    fns.sort()
    rf_fts=dict()
    for fn in fns:
        md_name=fn.split('_')[0]
        dummy_name=md_name+'_'+suffix_name+'.pkl'
        if fn!=dummy_name: continue

        pickle_path=os.path.join(folder,fn)
        with open(pickle_path,'rb') as f:
            data_dict=pickle.load(f)
        #avg_repr_grads=data_dict['avg_repr_grads']
        avg_embs_grads=data_dict['avg_embs_grads']
        #ft=np.concatenate([avg_embs_grads,avg_repr_grads],axis=0)
        #ft=avg_repr_grads
        ft=avg_embs_grads

        rf_fts[md_name]=ft
    return rf_fts

def read_pca_feature(folder, suffix_name):
    fns=os.listdir(folder)
    fns.sort()
    lr_fts=dict()
    for fn in fns:
        md_name=fn.split('_')[0]
        dummy_name=md_name+'_'+suffix_name+'.pkl'
        if fn!=dummy_name: continue

        pickle_path=os.path.join(folder,fn)
        with open(pickle_path,'rb') as f:
            data_dict=pickle.load(f)
        ft=data_dict['variance_ratio']

        lr_fts[md_name]=ft
    return lr_fts

def read_char_feature(folder, suffix_name):
    fns=os.listdir(folder)
    fns.sort()
    ch_fts=dict()
    for fn in fns:
        md_name=fn.split('_')[0]
        dummy_name=md_name+'_'+suffix_name+'.pkl'
        if fn!=dummy_name: continue

        pickle_path=os.path.join(folder,fn)
        with open(pickle_path,'rb') as f:
            data_dict=pickle.load(f)
        ft=data_dict['mis_ct']

        ch_fts[md_name]=ft
    return ch_fts


def read_data(gt_path, params):
    gt_dict=utils.read_gt_csv(gt_path)
    fts_list=list()
    for fn, folder, pattern in params:
        ft_dict=fn(folder, pattern)
        fts_list.append(ft_dict)

    arch_list=['GruLinear','LstmLinear','FCLinear']
    embe_list=['BERT','DistilBERT','GPT-2']
    arch_fets=list()
    features=[list() for _ in range(len(fts_list))]
    labels=list()
    names=list()
    for md_name in gt_dict:
        ok=True
        for ft_dict in fts_list:
            if md_name not in ft_dict:
                ok=False
                break
        if not ok: continue

        for k,ft_dict in enumerate(fts_list):
            ft=ft_dict[md_name].flatten()
            features[k].append(ft)

        lb=gt_dict[md_name]['poisoned']
        if lb=='True':
            lb=1
        else:
            lb=0
        labels.append(lb)
        names.append(md_name)

        k_ar=-1
        k_em=-1
        arch=gt_dict[md_name]['model_architecture']
        embe=gt_dict[md_name]['embedding']
        for k,ar in enumerate(arch_list):
            if ar==arch:
                k_ar=k
                break
        for k,em in enumerate(embe_list):
            if em==embe:
                k_em=k
                break
        if k_ar < 0 or k_em < 0:
            print(arch, embe)
            exit(0)
        n_arch=len(arch_list)
        n_embe=len(embe_list)
        arch_fet=np.zeros(n_arch+n_embe)
        arch_fet[k_ar]=1
        arch_fet[n_arch+k_em]=1
        arch_fets.append(arch_fet)

    arch_fets=np.asarray(arch_fets)
    features_list=[np.asarray(fts) for fts in features]
    #features_list[0] = np.concatenate([features_list[0],arch_fets],axis=1)
    labels=np.asarray(labels)

    return features_list, labels


if __name__=='__main__':

    home_root=os.environ['HOME']
    data_root=os.path.join(home_root,'data')

    gt_path=os.path.join(data_root,'round5-dataset-train-v0/METADATA.csv')
    v0_params=[gt_path, [(read_grads_feature, 'round5_rsts','v0_jacobian_normal_aug'),(read_pca_feature,'round5_rsts','v0_pca_clean')]]
    #v0_params=[gt_path, [(read_grads_feature, 'round5_rsts','v0_jacobian_normal_aug'),(read_pca_feature,'round5_rsts','v0_pca_clean'), (read_repr_feature,'round5_rsts','v0_jacobian_normal_aug')]]
    gt_path=os.path.join(data_root,'round5-dataset-train/METADATA.csv')
    #v1_params=[gt_path, [(read_grads_feature, 'round6_rsts','jacobian_normal_aug'),(read_pca_feature,'round6_rsts','pca_clean')]]
    v1_params=[gt_path, [(read_grads_feature, 'round5_rsts','jacobian_normal_aug'),(read_pca_feature,'round5_rsts','pca_clean')]]
    #v1_params=[gt_path, [(read_grads_feature, 'round5_rsts','jacobian_normal_aug'),(read_pca_feature,'round5_rsts','pca_clean'), (read_repr_feature,'round5_rsts','jacobian_normal_aug')]]
    gt_path=os.path.join(data_root,'new_models_round6_0/METADATA.csv')
    v2_params=[gt_path, [(read_grads_feature, 'new_model_round6_0_rsts','jacobian_normal_aug'),(read_pca_feature,'new_model_round6_0_rsts','pca_clean')]]
    gt_path=os.path.join(data_root,'new_models_round6_1/METADATA.csv')
    v3_params=[gt_path, [(read_grads_feature, 'new_model_round6_1_rsts','jacobian_normal_aug'),(read_pca_feature,'new_model_round6_1_rsts','pca_clean')]]

    dataset_params=[v0_params,v1_params,v2_params,v3_params]
    #dataset_params=[v1_params]


    clf_names=['RF','LogisticR']
    n_clf=len(clf_names)
    n_dataset=len(dataset_params)

    data_list=list()
    label_list=list()
    for gt_path, params in dataset_params:
        _fts_list, _lbs=read_data(gt_path, params)
        print('labels shape', _lbs.shape)
        data_list.append(_fts_list)
        label_list.append(_lbs)
    clf_fts_list=list()
    for k in range(n_clf):
        _list=list()
        for fts_list in data_list:
            _list.append(fts_list[k])
        ft_mat=np.concatenate(_list,axis=0)
        clf_fts_list.append(ft_mat)
    labels=np.concatenate(label_list,axis=0)

    print([fts.shape for fts in clf_fts_list])
    print(labels.shape)

    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier as RFC
    from sklearn.linear_model import LinearRegression as LR
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier as SGD
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from mlxtend.classifier import StackingCVClassifier, StackingClassifier
    from mlxtend.feature_selection import ColumnSelector
    from sklearn.metrics import roc_auc_score

    from demo_results import linear_adjust

    from lightgbm import LGBMClassifier

    #'''
    auc_list=list()
    rf_auc_list=list()
    best_test_acc=0
    kf=KFold(n_splits=10,shuffle=True)
    for train_index, test_index in kf.split(labels):

        Y_train, Y_test = labels[train_index], labels[test_index]

        #rf_clf=RFC(n_estimators=200, max_depth=11, random_state=1234)
        #rf_clf=RFC(n_estimators=200)
        rf_clf=LGBMClassifier(num_leaves=100)
        lr_clf=LogisticRegression(random_state=1234, max_iter=10000)
        repr_clf=RFC(n_estimators=100)
        clf_list=[rf_clf,lr_clf]
        stack_clf = LogisticRegression()
        #stack_clf = LGBMClassifier(num_leaves=1000)

        for fts,clf,na in zip(clf_fts_list, clf_list, clf_names):
            X_train, X_test = fts[train_index], fts[test_index]
            clf.fit(X_train,Y_train)

            preds=clf.predict(X_train)
            train_acc=np.sum(preds==Y_train)/len(Y_train)
            print(na+' train acc:', train_acc)

            score=clf.score(X_test, Y_test)
            preds=clf.predict(X_test)
            probs=clf.predict_proba(X_test)
            test_acc=np.sum(preds==Y_test)/len(Y_test)
            auc=roc_auc_score(Y_test, probs[:,1])
            print(na+' test acc:', test_acc, 'auc:',auc)
            if 'RF' in na:
                rf_auc_list.append(auc)

        probs_list_train=list()
        probs_list_test=list()
        for fts,clf,na in zip(clf_fts_list, clf_list, clf_names):
            X_train, X_test = fts[train_index], fts[test_index]
            probs_train=clf.predict_proba(X_train)
            probs_test=clf.predict_proba(X_test)
            probs_list_train.append(probs_train)
            probs_list_test.append(probs_test)
        probs_list_train=np.concatenate(probs_list_train,axis=1)
        probs_list_test=np.concatenate(probs_list_test,axis=1)

        stack_clf.fit(probs_list_train, Y_train)

        preds=stack_clf.predict(probs_list_train)
        train_acc=np.sum(preds==Y_train)/len(Y_train)
        print('>>>>  stack train acc:', train_acc)
        preds=stack_clf.predict(probs_list_test)
        probs=stack_clf.predict_proba(probs_list_test)
        test_acc=np.sum(preds==Y_test)/len(Y_test)
        auc=roc_auc_score(Y_test, probs[:,1])
        print('>>>>  stack test acc:', test_acc,'auc:',auc)

        sc_list=probs[:,1]
        lb_list=Y_test
        linear_adjust(lb_list,sc_list)

        auc_list.append(auc)
        if auc > max(auc_list)-1e-6:
            print('update auc to '+str(auc))
            import joblib
            joblib.dump(stack_clf,'stack_clf.joblib')
            joblib.dump(rf_clf,'rf_clf.joblib')
            joblib.dump(lr_clf,'lr_clf.joblib')


        print('============')

    rf_auc_list=np.asarray(rf_auc_list)
    print('RF AUC mean:',np.mean(rf_auc_list), 'std:',np.std(rf_auc_list),'max:',np.max(rf_auc_list),'min:',np.min(rf_auc_list))
    auc_list=np.asarray(auc_list)
    print('STACK AUC mean:',np.mean(auc_list), 'std:',np.std(auc_list),'max:',np.max(auc_list),'min:',np.min(auc_list))
    #'''

    #==========test=========
    import joblib
    rf_clf=joblib.load('rf_clf.joblib')
    lr_clf=joblib.load('lr_clf.joblib')
    stack_clf=joblib.load('stack_clf.joblib')
    clf_list=[rf_clf,lr_clf]

    probs_list=list()
    for fts,clf,na in zip(clf_fts_list, clf_list, clf_names):
        probs=clf.predict_proba(fts)
        probs_list.append(probs)
    probs_list=np.concatenate(probs_list,axis=1)
    stack_probs=stack_clf.predict_proba(probs_list)

    sc_list=stack_probs[:,1]
    lb_list=labels
    linear_adjust(lb_list, sc_list)








