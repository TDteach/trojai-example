import utils
import os
import pickle
import numpy as np

test_round='round5-dataset-train'
home_root=os.environ['HOME']
folder_root=os.path.join(home_root,'data',test_round)
gt_path=os.path.join(folder_root,'METADATA.csv')
feature_folder='rf_feature'


if __name__=='__main__':
    gt_dict=utils.read_gt_csv(gt_path)

    fns=os.listdir(feature_folder)
    fns.sort()
    features=list()
    labels=list()
    for fn in fns:
        if not fn.endswith('rf_feature.pkl') : continue
        md_name=fn.split('_')[0]

        lb=gt_dict[md_name]['poisoned']
        if lb=='True':
            lb=1
        else:
            lb=0
        labels.append(lb)

        pickle_path=os.path.join(feature_folder,fn)
        with open(pickle_path,'rb') as f:
            data_dict=pickle.load(f)
        avg_repr_grads=data_dict['avg_repr_grads']
        avg_embs_grads=data_dict['avg_embs_grads']
        ft=np.concatenate([avg_embs_grads,avg_repr_grads],axis=0)

        features.append(ft)
    features=np.asarray(features)
    labels=np.asarray(labels)

    print(features.shape)
    print(labels.shape)

    from sklearn.ensemble import RandomForestClassifier as RFC
    from sklearn.model_selection import KFold

    best_test_acc=0
    kf=KFold(n_splits=10,shuffle=True)
    for train_index, test_index in kf.split(features):
        X_train, Y_train = features[train_index], labels[train_index]
        X_test, Y_test = features[test_index], labels[test_index]

        clf=RFC(max_depth=10, random_state=1234)
        clf.fit(X_train,Y_train)

        preds=clf.predict(X_train)
        train_acc=np.sum(preds==Y_train)/len(Y_train)
        print('train acc:', train_acc)
        preds=clf.predict(X_test)
        test_acc=np.sum(preds==Y_test)/len(Y_test)
        print('test acc:', test_acc)

        if test_acc > best_test_acc:
            import joblib
            joblib.dump(clf,'rf.joblib')








