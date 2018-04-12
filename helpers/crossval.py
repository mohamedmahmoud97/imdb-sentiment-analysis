import numpy as np
import itertools

from sklearn.model_selection import cross_val_score,train_test_split

def cross_validate(X, y, classifier, parameters={},folds=None):
    
    n_samples = y.shape[0]
    perm_idx = np.random.permutation(n_samples)
    X_perm = X[perm_idx]
    y_perm = y[perm_idx]
    
    best_clf = None
    best_acc = 0
    clf_vars = []
    clf_mean_accs = []
    parameters_list = [{item[0]:item[1] for item in sublist} for 
                      sublist in itertools.product(*[[(key,val) for val in parameters[key]] 
                                                     for key in parameters.keys()])]
    
    
    for parameter_comb in parameters_list:
        clf = classifier(**parameter_comb)
        accs = cross_val_score(clf, X_perm, y_perm,cv=folds)
        mean_acc, var_acc = np.mean(accs), np.var(accs)
        clf_mean_accs.append(mean_acc)
        clf_vars.append(var_acc)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_clf = clf
        print(f'trained {clf}')
    
    return best_clf, clf_mean_accs, clf_vars

def tune_params(X_train,y_train,X_val,y_val,classifier,parameters={}):    
    best_clf = None
    best_acc = 0
    clf_accs = []
    parameters_list = [{item[0]:item[1] for item in sublist} for 
                      sublist in itertools.product(*[[(key,val) for val in parameters[key]] 
                                                     for key in parameters.keys()])]
    
    for parameter_comb in parameters_list:
        clf = classifier(**parameter_comb)
        clf.fit(X_train,y_train)
        clf_accs.append(np.mean(clf.predict(X_val) == y_val))
     
        if clf_accs[-1] > best_acc:
            best_acc = clf_accs[-1]
            best_clf = clf
        print(f'trained {clf}')
    
    return best_clf, clf_accs
  