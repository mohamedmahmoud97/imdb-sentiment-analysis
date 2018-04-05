# import pathlib
# from path import Path
# import numpy as np
# import os
# from gensim.models import Word2Vec,TfidfModel
# from gensim.corpora import Dictionary

import matplotlib.pyplot as plt
import seaborn
import numpy as np
import os
import nltk
import re
import gensim
from sklearn.model_selection import train_test_split
from helpers import *
from cleaners import *
from tfidf_w2v_helpers import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pathlib
from multiprocessing import Process
from pathlib import Path
from gensim.models import Word2Vec,TfidfModel,Phrases
from gensim.corpora import Dictionary
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier, Lasso
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import inspect
import itertools
from threading import Thread,Lock
from sklearn.neighbors import KNeighborsClassifier

# an iterator used by gensim to train models without loading the whole corpus in memory
class MyDocs(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        pathlist = Path(self.dirname).glob('**/*')
        for path in pathlist:
            path_in_str = str(path)
            if os.path.isfile(path_in_str):
                f=open(path_in_str)
                x=f.read().split()
                yield x


def word2vec(docs_root,model_name=None):
    # Training a word2vec model, the bigram transformer tells gensim to create vectors for bigram phrases such New York
    save = model_name!=None
    model_path = f'models/wv_{model_name}'
    model_exists = os.path.isfile(model_path)
    dir_exists = os.path.isdir('models')

    wv_model = None
    
    if model_exists:
        wv_model = Word2Vec.load(model_path)
    else:
        docs = MyDocs(docs_root)
        wv_model = Word2Vec(docs, size=300, window=5, min_count=100, workers=6)
    
    if save:
        if not dir_exists:
            pathlib.Path.mkdir('models')
        if not model_exists:
            wv_model.save(model_path)

    return wv_model


def tf_idf_model(doc_path,model_name=None):
    model_path = f'models/tf_{model_name}'
    dict_path = f'models/dict_{model_name}'
    model_exists = os.path.isfile(model_path)
    dict_exists = os.path.isfile(dict_path)
    dir_exists = os.path.isdir('models')
    save = model_name!=None


    docs = MyDocs(doc_path)
    bow_dict = Dictionary(docs)
    corpus = [bow_dict.doc2bow(doc) for doc in docs]
    tf_model = TfidfModel(corpus)
    if save:
        if not dir_exists:
            pathlib.Path.mkdir('models')
        if not model_exists:
            tf_model.save(model_path)
        if not dict_exists:
            bow_dict.save(dict_path)
    return corpus,bow_dict,tf_model
        

def tf_doc2vec_factory(bow_dict,tf_model,word2vec):
    def tf_doc2vec(doc_bow,raw_doc):
        tf_dict = {e0:e1 for e0,e1 in tf_model[doc_bow]}
        doc_tf_idf = np.array([tf_dict[tok_id] for tok_id in (bow_dict.token2id[tok] for tok in raw_doc if tok in word2vec.wv.vocab)])
        vecs = np.array([word2vec.wv[word] for idx,word in enumerate(raw_doc) if word in word2vec.wv.vocab])
        doc_vec = np.sum(doc_tf_idf.reshape(-1,1)*vecs,axis=0)
        return doc_vec
    return tf_doc2vec


def tf_wv_samples(model_prefix,root_dir,wv=None):
    # root dir is assumed to have children /pos and /neg
    # word 2 vec model is created using all descendents of the parent of root_dir
    # returns X,y 
    pos_dir = f'{root_dir}/pos'
    neg_dir = f'{root_dir}/neg'
    
    # get the word2vec model
    if wv is None:
        wv = word2vec(root_dir,model_prefix)
        
    # get tf-idf models
    pos_corpus, pos_dict, pos_tf_model = tf_idf_model(pos_dir, f'{model_prefix}_pos')
    neg_corpus, neg_dict, neg_tf_model = tf_idf_model(neg_dir, f'{model_prefix}_neg')
    # get tf-doc2vec factories
    pos_tf_doc2vec = tf_doc2vec_factory(pos_dict, pos_tf_model, wv)
    neg_tf_doc2vec = tf_doc2vec_factory(neg_dict, neg_tf_model, wv)
    
    pos_vecs = []
    for bows,doc in zip(pos_corpus,MyDocs(pos_dir)):
        pos_vecs.append(pos_tf_doc2vec(bows,doc))
    pos_vecs = np.array(pos_vecs)
    
    neg_vecs = []
    for bows,doc in zip(neg_corpus,MyDocs(neg_dir)):
        neg_vecs.append(neg_tf_doc2vec(bows,doc))
    neg_vecs = np.array(neg_vecs)
    
    X = np.concatenate((pos_vecs,neg_vecs),axis=0)
    y = np.asarray([1]*pos_vecs.shape[0] + [-1]*pos_vecs.shape[0])
    
    return X, y, wv
