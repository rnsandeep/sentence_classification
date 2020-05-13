import pandas as pd
import warnings
import numpy as np
import string
import pickle
from tqdm import tqdm

import gensim
from gensim.models import Word2Vec

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from dateutil.parser import parse


def lower_case(x):
    return [x1.lower() for x1 in x.split() if not isinstance(x1, float)] if not isinstance(x, float) else [x]

def remove_punctuation(x):
    return [x1.translate(str.maketrans("", "", string.punctuation)) for x1 in x if not isinstance(x1, float)]

stop_words = set(stopwords.words('english'))

def remove_stop_words(x):
    filtered_sentence = [w for w in x if not w in stop_words]
    return filtered_sentence

ps = PorterStemmer()

def stemming(x):
    return [ps.stem(w) for w in x]

def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)
    if not mean:
        return np.zeros(wv.vector_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in tqdm(text_list) ])

def preprocessSentences(file_name):
    df = pd.read_csv(file_name, sep='\t')
    df = df[~df['title'].isna()]
    df['title_lower'] = df['title'].apply(lambda x:lower_case(x))
    stop_words = set(stopwords.words('english'))
    df['title_punc'] = df['title_lower'].apply(lambda x:remove_punctuation(x))
    df['stop_removal'] = df['title_punc'].apply(lambda x:remove_stop_words(x))
    df['stemming'] = df['stop_removal'].apply(lambda x:stemming(x))

    all_sentence = df['stemming']
    return all_sentence

def getWord2VecFeatures(file_name):
    model = pickle.load(open('word2vec_title_model.pkl','rb'))
    all_sentence = preprocessSentences(file_name)
    all_features = word_averaging_list(model, all_sentence)
    return all_features
