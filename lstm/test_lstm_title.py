import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import pickle

df = pd.read_csv("validation.csv", sep='\t')

df = df.applymap(str)

X_test = df['title']


model, le, tok = pickle.load(open('model_title.pkl','rb'))


max_words = 1000
max_len = 150


#model, le, tok = pickle.load(open('model.pkl','rb'))

Y = df['category']

#le = LabelEncoder()
Y = le.fit_transform(Y)
Y_test = Y.reshape(-1,1)


#pickle.dump(model, open('model.pkl','wb'))

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

print(model.predict(test_sequences_matrix))
accr = model.evaluate(test_sequences_matrix, Y_test)


print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
