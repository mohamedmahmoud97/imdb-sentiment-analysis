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
