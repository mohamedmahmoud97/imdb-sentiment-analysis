import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pathlib
from multiprocessing import Process,Lock,Value

def speedy(dirs,n_jobs=4,target_suffix='clean'):
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
            proc = Process(target=write_clean_docs, args=(curr_dir,target_dir,start,end,))
            procs.append(proc)
            proc.start()
    for proc in procs:
        proc.join()
    return curr_dir.split('/')[0]+f'_{target_suffix}/'
    

def dir_size(src_dir):
    return len([name for name in os.listdir(os.fsencode(src_dir))])

def write_clean_docs(src_dir,target_dir,start,end):
    
    directory = os.fsencode(src_dir)
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if( start <= int(filename.split('_')[0]) <= end):
            with open(os.path.join(src_dir,filename),'r') as f:
                current_doc = f.read()
            with open(os.path.join(target_dir,filename),'w') as f:
                f.write(clean_doc(current_doc))

def clean_doc(doc):
    lemmatizer = WordNetLemmatizer()
    
    clean_doc = re.sub(r'<.+?>|[!"#$%&\'()=*+,-./:;?@\[\]^_`{|}~<>]|[0-9]', ' ', doc)
    clean_doc = ' '.join([word.lower() for word in clean_doc.split()])
    tokens =  [word for word in clean_doc.split() if word not in set(stopwords.words('english'))]
    tagged_tokens = [(pair[0],get_wordnet_pos(pair[1])) for pair in nltk.pos_tag(tokens)]
    clean_doc = ' '.join([ lemmatizer.lemmatize(word,tag) for word,tag in tagged_tokens])
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