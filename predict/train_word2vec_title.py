import pandas as pd
import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import gensim
from gensim.models import Word2Vec
import numpy as np
import string
import pickle
from tqdm import tqdm
from nltk.corpus import stopwords 
from sklearn.linear_model import LogisticRegression

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
    return np.vstack([word_averaging(wv, post) for post in tqdm(text_list)])

def lower_case(x):
    return [x1.lower() for x1 in x.split() if not isinstance(x1, float)] if not isinstance(x, float) else [x]

def remove_stop_words(x):
    filtered_sentence = [w for w in x if not w in stop_words] 
    return filtered_sentence

def remove_punctuation(x):
    return [x1.translate(str.maketrans("", "", string.punctuation)) for x1 in x if not isinstance(x1, float)]

from nltk.stem import PorterStemmer
ps = PorterStemmer() 

def stemming(x):
    return [ps.stem(w) for w in x]

def preprocessSentences(file_name):
    df = pd.read_csv(file_name, sep='\t')
    df = df[~df['title'].isna()]
    df['title_lower'] = df['title'].apply(lambda x:lower_case(x))
    stop_words = set(stopwords.words('english'))
    df['title_punc'] = df['title_lower'].apply(lambda x:remove_punctuation(x))
    df['stop_removal'] = df['title_punc'].apply(lambda x:remove_stop_words(x))
    df['stemming'] = df['stop_removal'].apply(lambda x:stemming(x))
    all_sentence = df['stemming']
    labels = df['category']
    return all_sentence, labels

def getWord2vecTrainFeatures(file_name):

    all_sentence, train_labels = preprocessSentences(file_name)
    model_word2vec = gensim.models.Word2Vec(all_sentence, min_count = 1,
                              size = 200, window = 5)
    all_features = word_averaging_list(model_word2vec.wv, all_sentence)
    return all_features, model_word2vec, train_labels

def getWord2VecFeatures(file_name, model_word2vec):
    all_sentence, test_labels = preprocessSentences(file_name)
    all_features = word_averaging_list(model_word2vec.wv, all_sentence)
    return all_features, test_labels

train_features, word2vec_model, train_labels = getWord2vecTrainFeatures('training.tsv')
test_features, test_labels = getWord2VecFeatures('test1.csv', word2vec_model)

clf = LogisticRegression(random_state=0).fit(train_features, train_labels)

predictions = clf.predict(test_features)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))


