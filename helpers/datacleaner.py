import numpy as np
import nltk
import re

from pathlib import Path
from threading import Thread,Lock

from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer


from gensim.models import Word2Vec,TfidfModel,Phrases
from gensim.corpora import Dictionary

cachedStopWords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def basic_cleaner(doc):
    global cachedStopWords
    global lemmatizer
    clean_doc = re.sub(r'<.+?>|[!"#$%&\'()=*+,-./:;?@\[\]^_`{|}~<>]|[0-9]', ' ', doc)
    clean_doc = ' '.join([word.lower() for word in clean_doc.split()])
    tokens =  [word for word in clean_doc.split() if word not in cachedStopWords]
    tagged_tokens = [(pair[0],get_wordnet_pos(pair[1])) for pair in nltk.pos_tag(tokens)]
    clean_doc = ' '.join([ lemmatizer.lemmatize(word,tag) for word,tag in tagged_tokens])
    return clean_doc

def pos_cleaner(doc):
    global lemmatizer
    global cachedStopWords
    
    clean_doc = re.sub(r'<.+?>|[!"#$%&\'()=*+,-./:;?@\[\]^_`{|}~<>]|[0-9]', ' ', doc)
    clean_doc = ' '.join([word.lower() for word in clean_doc.split()])
    tokens =  [word for word in clean_doc.split() if word not in cachedStopWords]
    clean_doc = ' '.join([ lemmatizer.lemmatize(word,get_wordnet_pos(tag))+'_'+tag for word,tag in nltk.pos_tag(tokens)])
    
    return clean_doc

# # Multithreaded Cleaning
# def batch_cleaner(docs, clean_docs, cleaner, lock, start, end):
#     my_cleaned_docs = []
#     for i in range(start,end):
#         my_cleaned_docs.append(cleaner(docs[i]))

#     lock.acquire()
#     clean_docs.extend(my_cleaned_docs)
#     print(f'cleaned docs [{start},{end}]')
#     lock.release()
  
# def speedy(cleaner,docs,n_jobs=4):
#     clean_docs = []
#     threads = []
#     n_docs = len(docs)
#     lock = Lock()
#     for i in range(n_jobs):
#         start = i*n_docs//n_jobs
#         end = (i+1)*n_docs//n_jobs
#         threads.append(Thread(target=batch_cleaner, args=(docs, clean_docs, cleaner, lock, start, end )))
#         threads[-1].start()
        
#     for thread in threads:
#         thread.join()

#     return clean_docs