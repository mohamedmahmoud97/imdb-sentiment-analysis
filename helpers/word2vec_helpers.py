import numpy as np

from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer


from gensim.models import Word2Vec,TfidfModel,Phrases
from gensim.corpora import Dictionary

def word2vec(docs,model_name=None):
    doc_tokens = [doc.split() for doc in docs]
    wv_model = Word2Vec(doc_tokens, size=300, window=5, min_count=50, workers=10)
    return wv_model

def tf_idf_model(docs,model_name=None):

    doc_tokens = [doc.split() for doc in docs]
    bow_dict = Dictionary(doc_tokens)
    corpus = [bow_dict.doc2bow(doc) for doc in doc_tokens]
    tf_model = TfidfModel(corpus)
    
    return bow_dict,tf_model

def tf_doc2vec_factory(bow_dict,tf_model,word2vec):
    def tf_doc2vec(raw_doc):
        doc_tokens = [token for token in raw_doc.split() if token in word2vec.wv.vocab]
        tf_dict = {e0:e1 for e0,e1 in tf_model[bow_dict.doc2bow(doc_tokens)]}
        doc_tf_idf = np.array([tf_dict[tok_id] for tok_id in bow_dict.doc2idx(doc_tokens)])
        vecs = np.array([word2vec.wv[token] for token in doc_tokens])
        doc_vec = np.sum(doc_tf_idf.reshape(-1,1)*vecs,axis=0)
        return doc_vec
    return tf_doc2vec

def tf_wv_samples(pos_docs, neg_docs, model_prefix, wv=None):

    all_docs = np.asarray(pos_docs + neg_docs)
    np.random.shuffle(all_docs)
    
    
    # get the word2vec model
    if wv is None:
        wv = word2vec(all_docs,model_prefix)
        
    # get tf-idf models
    pos_dict, pos_tf_model = tf_idf_model(pos_docs, f'{model_prefix}_pos')
    neg_dict, neg_tf_model = tf_idf_model(neg_docs, f'{model_prefix}_neg')
    # get tf-doc2vec factories
    pos_tf_doc2vec = tf_doc2vec_factory(pos_dict, pos_tf_model, wv)
    neg_tf_doc2vec = tf_doc2vec_factory(neg_dict, neg_tf_model, wv)
    
    pos_vecs = []
    for doc in pos_docs:
        pos_vecs.append(pos_tf_doc2vec(doc))
    pos_vecs = np.array(pos_vecs)
    
    neg_vecs = []
    for doc in neg_docs:
        neg_vecs.append(neg_tf_doc2vec(doc))
    neg_vecs = np.array(neg_vecs)
    
    X = np.concatenate((pos_vecs,neg_vecs),axis=0)
    y = np.asarray([1]*pos_vecs.shape[0] + [-1]*pos_vecs.shape[0])
    return X, y, wv

