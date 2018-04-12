import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pathlib
from multiprocessing import Process,Lock,Value
import itertools
import numpy as np
from sklearn.model_selection import cross_val_score


def speedy(cleaner, dirs,n_jobs=4,target_suffix='clean'):
    procs = []
    procs_per_dir = n_jobs

    for curr_dir in dirs:
        curr_dir_size = dir_size(curr_dir)
        target_dir = curr_dir.split('/')[0]+f'_{target_suffix}/'+'/'.join(curr_dir.split('/')[1:])
        if(os.path.isdir(target_dir)):
            continue
        print(f'cleaning{curr_dir}')
        for i in range(procs_per_dir):
            start = i*curr_dir_size//procs_per_dir
            end = (i+1)*curr_dir_size//procs_per_dir
            proc = Process(target=write_clean_docs, args=(cleaner,curr_dir,target_dir,start,end,))
            procs.append(proc)
            proc.start()
    for proc in procs:
        proc.join()
    return curr_dir.split('/')[0]+f'_{target_suffix}/'
    

def dir_size(src_dir):
    return len([name for name in os.listdir(os.fsencode(src_dir))])

def write_clean_docs(cleaner,src_dir,target_dir,start,end):
    
    directory = os.fsencode(src_dir)
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if( start <= int(filename.split('_')[0]) <= end):
            with open(os.path.join(src_dir,filename),'r') as f:
                current_doc = f.read()
            with open(os.path.join(target_dir,filename),'w') as f:
                f.write(cleaner(current_doc))


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
        accs = cross_val_score(clf, X_perm, y_perm, n_jobs=-1,cv=folds)
        mean_acc, var_acc = np.mean(accs), np.var(accs)
        clf_mean_accs.append(mean_acc)
        clf_vars.append(var_acc)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_clf = clf
        print(f'trained {clf}')
    
    return best_clf, clf_mean_accs, clf_vars