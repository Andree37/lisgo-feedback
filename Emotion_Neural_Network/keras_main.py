#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:33:53 2020

@author: andre
"""
import numpy as np


import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Embedding,Flatten
from keras.layers import LSTM
from keras.models import save_model
from matplotlib import pyplot
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
from keras.layers import Conv1D,MaxPooling1D
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
import time
import datetime
import pickle

MAX_SEQUENCE_LENGTH=20
EMBEDDING_DIM=50

stop_words = set(stopwords.words('english')) 
new_stop_words=set(stop_words)



dataFrame=pd.read_csv('text_emotion_twitter.csv', encoding='utf-8')

x=dataFrame.values[:,3]
y=dataFrame.values[:,1]

# x=only the tweet
# y=sentiment


print(x.shape)
print(y.shape)


# Preprocessing Text Data

# adding woudlnt type of words into stopwords list
for s in stop_words:
	new_stop_words.add(s.replace('\'',''))
	pass
	
stop_words=new_stop_words

# removing @ from the text and the thing that follows, since it might be the user
base_filters='\n\t!"#$%&()*+,-./:;<=>?[\]^_`{|}~ '

word_sequences=[]

for i in x:
	
	i=str(i)

	i=i.replace('\'', '')
	newlist = [x for x in text_to_word_sequence(i,filters=base_filters, lower=True) if not x.startswith("@")]
	filtered_sentence = [w for w in newlist if not w in stop_words] 
	word_sequences.append(filtered_sentence)



# Tokenizing words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(word_sequences)
word_indices = tokenizer.texts_to_sequences(word_sequences)
word_index = tokenizer.word_index
print("Tokenized to Word indices as ")
print(np.array(word_indices).shape)


# padding word_indices
x_data=pad_sequences(word_indices,maxlen=MAX_SEQUENCE_LENGTH)
print("After padding data")
print(x_data.shape)


# Building Embedding Layer

# using pretrained glove vector from kaggle
print("Loading Glove Vectors ...")
embeddings_index = {}
f = open(os.path.join('', 'glove.6B.50d.txt'),'r',encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Loaded GloVe Vectors Successfully')



embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print("Embedding Matrix Generated : ",embedding_matrix.shape)



embedding_layer = Embedding(len(word_index) + 1,EMBEDDING_DIM, weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=False)


# Encoding Labels 
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
le_name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_),label_encoder.classes_))
print("Label Encoding Classes as ")
print(le_name_mapping)

y_data=np_utils.to_categorical(integer_encoded)
print("One Hot Encoded class shape ")
print(y_data.shape)


# Building Model 
model=Sequential()
model.add(embedding_layer)
model.add(Conv1D(30,1,activation="relu"))
model.add(MaxPooling1D(4))
model.add(LSTM(100,return_sequences=True))
model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dense(y_data.shape[1],activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
print(model.summary())

print("Finished Preprocessing data ...")
print("x_data shape : ",x_data.shape)
print("y_data shape : ",y_data.shape)

# spliting data into training, testing set
print("spliting data into training, testing set")
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data)


batch_size = 64
num_epochs = 100
x_valid, y_valid = x_train[:batch_size], y_train[:batch_size]
x_train2, y_train2 = x_train[batch_size:], y_train[batch_size:]


st=datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
# define the checkpoint
# save to file
filepath="model_weights-improvement-{epoch:02d}-{accuracy:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history=model.fit(x_train2, y_train2, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=num_epochs,callbacks=callbacks_list)

scores = model.evaluate(x_test, y_test, verbose=0)
save_model(model, "best_model.hdf5", overwrite=True, include_optimizer=True)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Test accuracy:', scores[1])

pyplot.plot(history.history['val_accuracy'],label='Training Accuracy')
pyplot.plot(history.history['accuracy'],label='Validation Accuracy')

pyplot.legend()
pyplot.show()
