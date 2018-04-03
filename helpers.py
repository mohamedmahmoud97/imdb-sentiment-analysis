import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pathlib
from multiprocessing import Process,Lock,Value

def speedy(dirs,n_jobs=4):
    procs = []
    procs_per_dir = n_jobs
    lock = Lock()

    for (idx,curr_dir) in enumerate(dirs):
        curr_dir_size = dir_size(curr_dir)
        cleaned_files = [0]
        for i in range(procs_per_dir):
            start = i*curr_dir_size//procs_per_dir
            end = (i+1)*curr_dir_size//procs_per_dir
            proc = Process(target=write_clean_docs, args=(curr_dir,start,end,lock,(i+1)*(idx+1),curr_dir_size,cleaned_files))
            procs.append(proc)
            proc.start()
    for proc in procs:
        proc.join()

def dir_size(src_dir):
    print(src_dir)
    return len([name for name in os.listdir(os.fsencode(src_dir))])

def write_clean_docs(src_dir,start,end,lock,p_num=-1,total_files=100,cleaned_files=None):
    
    directory = os.fsencode(src_dir)
    target_dir = src_dir.split('/')[0]+'_clean/'+'/'.join(src_dir.split('/')[1:])
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if( start <= int(filename.split('_')[0]) <= end):
            with open(os.path.join(src_dir,filename),'r') as f:
                current_doc = f.read()
            with open(os.path.join(target_dir,filename),'w') as f:
                f.write(clean_doc(current_doc))
                
            lock.acquire()
            cleaned_files[0] +=1
            print(f'{p_num} | {src_dir}: progress => {cleaned_files[0]}/{total_files} ')
            lock.release()

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