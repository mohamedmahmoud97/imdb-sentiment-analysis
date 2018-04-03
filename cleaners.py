import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet



def basic_cleaner(doc):
    lemmatizer = WordNetLemmatizer()
    
    clean_doc = re.sub(r'<.+?>|[!"#$%&\'()=*+,-./:;?@\[\]^_`{|}~<>]|[0-9]', ' ', doc)
    clean_doc = ' '.join([word.lower() for word in clean_doc.split()])
    tokens =  [word for word in clean_doc.split() if word not in set(stopwords.words('english'))]
    tagged_tokens = [(pair[0],get_wordnet_pos(pair[1])) for pair in nltk.pos_tag(tokens)]
    clean_doc = ' '.join([ lemmatizer.lemmatize(word,tag) for word,tag in tagged_tokens])
    return clean_doc

def pos_cleaner(doc):
    lemmatizer = WordNetLemmatizer()   
    clean_doc = re.sub(r'<.+?>|[!"#$%&\'()=*+,-./:;?@\[\]^_`{|}~<>]|[0-9]', ' ', doc).lower()
    tokens =  [word for word in clean_doc.split() if word not in set(stopwords.words('english'))]
    clean_doc = ' '.join([ lemmatizer.lemmatize(word,get_wordnet_pos(tag))+'_'+tag for word,tag in nltk.pos_tag(tokens)])
    return clean_doc


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