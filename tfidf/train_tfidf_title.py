import pandas as pd
import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
#import gensim
#from gensim.models import Word2Vec
import numpy as np
import string
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


stop_words = set(stopwords.words('english'))

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

all_sentence, train_labels = preprocessSentences('training.tsv')
all_sentence = [' '.join(x) for x in all_sentence]


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(all_sentence)

all_sentence, test_labels = preprocessSentences('validation.csv')
all_sentence = [' '.join(x) for x in all_sentence]

test_vectors = vectorizer.transform(all_sentence)


#clf = LinearSVC(random_state=0, tol=1e-5)

#clf.fit(train_vectors, train_labels)

#predictions = clf.predict(test_vectors)

clf = LogisticRegression(random_state=0).fit(train_vectors, train_labels)

predictions = clf.predict(test_vectors)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))


