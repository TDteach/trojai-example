import utils
import os
import pickle
import numpy as np

test_round='round5-dataset-train'
home_root=os.environ['HOME']
folder_root=os.path.join(home_root,'data',test_round)
gt_path=os.path.join(folder_root,'METADATA.csv')
grads_feature_folder='rf_feature'
#grads_name_pattern='jacobin_feature'
grads_name_pattern='jacobin_split_normal_feature'
pca_feature_folder='round5_rsts'
pca_name_pattern='pca.pkl'


if __name__=='__main__':
    gt_dict=utils.read_gt_csv(gt_path)

    #============grads feature extraction==========
    fns=os.listdir(grads_feature_folder)
    fns.sort()
    rf_fts=dict()
    for fn in fns:
        if not grads_name_pattern in fn : continue
        md_name=fn.split('_')[0]

        pickle_path=os.path.join(grads_feature_folder,fn)
        with open(pickle_path,'rb') as f:
            data_dict=pickle.load(f)
        avg_repr_grads=data_dict['avg_repr_grads']
        avg_embs_grads=data_dict['avg_embs_grads']
        #ft=np.concatenate([avg_embs_grads,avg_repr_grads],axis=0)
        #ft=avg_repr_grads
        ft=avg_embs_grads

        rf_fts[md_name]=ft
        print(ft.shape)
        exit(0)


    #============pca feature extraction==========
    fns=os.listdir(pca_feature_folder)
    fns.sort()
    lr_fts=dict()
    z = 0
    for fn in fns:
        if not pca_name_pattern in fn : continue
        if 'double' in fn : continue
        if 'clean' in fn : continue
        md_name=fn.split('_')[0]

        pickle_path=os.path.join(pca_feature_folder,fn)
        with open(pickle_path,'rb') as f:
            data_dict=pickle.load(f)
        ft=data_dict['variance_ratio']

        lr_fts[md_name]=ft

        z+=1

    clf_fts_list=[rf_fts,lr_fts]
    clf_fts_idx=None
    features=[list() for _ in range(len(clf_fts_list))]
    labels=list()
    names=list()
    for md_name in gt_dict:
        ok=True
        for fts in clf_fts_list:
            if md_name not in fts:
                ok=False
                break
        if not ok: continue

        for k,fts in enumerate(clf_fts_list):
            ft=fts[md_name].flatten()
            features[k].append(ft)

        lb=gt_dict[md_name]['poisoned']
        if lb=='True':
            lb=1
        else:
            lb=0
        labels.append(lb)
        names.append(md_name)

    features=[np.asarray(fts) for fts in features]
    labels=np.asarray(labels)


    print([fts.shape for fts in features])
    print(labels.shape)

    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier as RFC
    from sklearn.linear_model import LinearRegression as LR
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier as SGD
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from mlxtend.classifier import StackingCVClassifier, StackingClassifier
    from mlxtend.feature_selection import ColumnSelector
    from sklearn.metrics import roc_auc_score


    rf_clf=RFC(n_estimators=100, max_depth=8, random_state=1234)
    #lr_clf=SGD(random_state=1234, max_iter=10000)
    lr_clf=LogisticRegression(random_state=1234, max_iter=10000)
    #lr_clf = SVC(random_state=0, probability=True)
    #stack_clf = RFC(max_depth=5, random_state=1234)
    stack_clf = LogisticRegression()

    clf_list=[rf_clf,lr_clf]
    clf_names=['RF','LogisticR']


    auc_list=list()
    rf_auc_list=list()
    best_test_acc=0
    kf=KFold(n_splits=10,shuffle=True)
    for train_index, test_index in kf.split(labels):

        Y_train, Y_test = labels[train_index], labels[test_index]

        for fts,clf,na in zip(features, clf_list, clf_names):
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
        for fts,clf,na in zip(features, clf_list, clf_names):
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

        auc_list.append(auc)
        if auc > max(auc_list)-1e-6:
            import joblib
            joblib.dump(stack_clf,'stack_clf.joblib')
            joblib.dump(rf_clf,'rf_clf.joblib')
            joblib.dump(lr_clf,'lr_clf.joblib')


        print('============')

    rf_auc_list=np.asarray(rf_auc_list)
    print('RF AUC mean:',np.mean(rf_auc_list), 'std:',np.std(rf_auc_list),'max:',np.max(rf_auc_list),'min:',np.min(rf_auc_list))
    auc_list=np.asarray(auc_list)
    print('STACK AUC mean:',np.mean(auc_list), 'std:',np.std(auc_list),'max:',np.max(auc_list),'min:',np.min(auc_list))








