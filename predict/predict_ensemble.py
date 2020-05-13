from sklearn.linear_model import LogisticRegression
import pickle
import sys
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence 

from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, classification_report
from get_word2vec_title import getWord2VecFeatures

stop_words = set(stopwords.words('english'))


def getLstmFeatures(le, tok, file_name):
    df = pd.read_csv(file_name, sep='\t')
    df = df.applymap(str)
    X_test = df['description']
    labels = df['category']
    max_len = 150
    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)
    return test_sequences_matrix, labels 


file_name = sys.argv[1]

df = pd.read_csv(file_name, sep='\t')
model, le, tok = pickle.load(open('lstm_description_best.pkl','rb'))  # load the best lstm model for descriptions.
test_sequences_matrix, test_labels = getLstmFeatures(le, tok, file_name) # get lstm features.
descp_probs = model.predict(test_sequences_matrix)

submission = pd.DataFrame()

submission['title'] = df['title']
submission['description'] = df['description']
descriptions = [str(i) for i in df['description'].values]

submission['category'] = df['category']

logistic_clf = pickle.load(open('logistic_title_best.pkl','rb'))
test_title = getWord2VecFeatures(file_name)
title_probs = logistic_clf.predict_proba(test_title)

predictions = []
labels = ['R', 'S']


# we combinely check the probabilities of title predictions and description predictions and assign the label.

i = 0
for l, d in zip( title_probs, descp_probs):

    #check if the description is empty if yes then assign the label of title. 
    if not pd.isna(submission['description'].iloc[i]) and l[np.argmax(l)] < 0.6:
        d = 0 if d > 0.5 else 1
        predictions.append(labels[d])
        print(labels[d])
    else:
        predictions.append(labels[np.argmax(l)])
    i = i + 1

submission['predictions'] = predictions

submission.to_csv("submission.tsv", sep='\t')

print("confusion matrix")
print(confusion_matrix(test_labels, predictions))
print("classification report")
print(classification_report(test_labels, predictions))
