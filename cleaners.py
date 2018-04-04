import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet



def basic_cleaner(doc):
    lemmatizer = WordNetLemmatizer()
    clean_doc = re.sub('<.+?>',' ',doc).lower()
    clean_doc = re.sub("'m",' am',clean_doc)
    clean_doc = re.sub("'re",' are',clean_doc)
    clean_doc = re.sub("'s",' is',clean_doc)
    clean_doc = re.sub("n't",' not',clean_doc)
    clean_doc = re.sub("'ll",' will',clean_doc) 
    clean_doc = re.sub("[!?(),`'\"./:;%$@#]"," ",clean_doc)
      
       
    tokens = [token for token in nltk.word_tokenize(clean_doc) if token not in set(stopwords.words('english')) and not token.isdigit()]
    clean_doc = ' '.join([lemmatizer.lemmatize(token,get_wordnet_pos(tag)) for token,tag in nltk.pos_tag(tokens)])
    return clean_doc

def pos_cleaner(doc):
    lemmatizer = WordNetLemmatizer()   
    clean_doc = re.sub('<.+?>',' ',doc).lower()
    tokens = [token for token in nltk.word_tokenize(clean_doc) if token not in set(stopwords.words('english'))]
    clean_doc = ' '.join([lemmatizer.lemmatize(token,get_wordnet_pos(tag))+'_'+tag for token,tag in nltk.pos_tag(tokens)])
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