#################################################################################
## BEST TO RUN THIS FILE IS IN AN AWS EC2 GPU ENABLED INSTANCE WITH LARGE DATASET.
#################################################################################

# -*- coding: utf-8 -*-
"""
"""
#print('install gensim')
#!pip install gensim
#print('gensim installed, import packages')
import json
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json

import sys
import random
import os

#########################################
## Uncomment in Colab
#########################################
# from google.colab import drive
# drive.mount('/content/drive')

src_path = os.environ['PWD']

#########################################
## Change the paths when outside of Colab
#########################################




##########################################################################################
## all_data_for_embedding_matrix replaced with word embeddings.vocab for this iteration. 
## commented out for potential inclusion in later iteration
##########################################################################################
## all_data_for_embedding_matrix = pd.read_csv('whole_corpus_for_embedding_matrix.csv', 
##                                            header=None, names=['ad_creative_body'], dtype='str',
##                                            encoding='utf-8')
## all_data_for_embedding_matrix = all_data_for_embedding_matrix.ad_creative_body
## all_data_for_embedding_matrix = all_data_for_embedding_matrix.map(str)
##########################################################################################
labeled_data = pd.read_csv('labeled_data.csv'
                           #, dtype='str', encoding='utf-8'
                          );


########################################
## set directories and parameters
########################################
#BASE_DIR = src_path+'/' 
EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin.gz'
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 3000000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
EPOCHS = 500
BATCH_SIZE = 1024
SAMP_SIZE = len(labeled_data)-1
T_T_CUTOFF = int(round(SAMP_SIZE*.7))

print('past variable declaration stage')
num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)
########################################
## index word vectors
########################################

print('Indexing word vectors, this will take a few minutes.')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))
print('we did it!  Word embeddings created')
rndm_sample = labeled_data.sample(SAMP_SIZE, random_state=4125)

print('creating random samples for test')
X_ = rndm_sample.ad_creative_body
y_ = rndm_sample.lean
labels = np.array(y_)
print('random sampling finished - now tokenize')

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(list(word2vec.vocab.keys()))
sequence_1 = tokenizer.texts_to_sequences(X_)
word_index = tokenizer.word_index
print('Found %s unique tokens, now start padding' % len(word_index))

### Notice that sequences are of varying lengths. solution: padding 

X_seq = pad_sequences(sequence_1, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
print('Shape of data tensor:', X_seq.shape)


print('Preparing embedding matrix')
nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    #print(word, i)
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

X_seq[0]

data_1_train = X_seq[:T_T_CUTOFF]
labels_train = y_[:T_T_CUTOFF]

data_1_test = X_seq[T_T_CUTOFF:]
labels_test = y_[T_T_CUTOFF:]

weight_val = np.ones(len(labels_test))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_test==0] = 1.309028344

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)

lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)


x1 = Dropout(rate_drop_dense)(x1)
x1 = BatchNormalization()(x1)

x1 = Dense(num_dense, activation=act)(x1)
x1 = Dropout(rate_drop_dense)(x1)
x1 = BatchNormalization()(x1)

preds = Dense(1, activation='sigmoid')(x1)

########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

model = Model(inputs=sequence_1_input, \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
print(model.summary())
print(STAMP)

########################################
## train the model
########################################


early_stopping = EarlyStopping(monitor='val_loss', patience=99)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit(data_1_train, labels_train, \
        validation_data=(data_1_test, labels_test, weight_val), \
        epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

#######################################
## Change the path when outside of colab
#######################################

#!/usr/bin/env python
import logging
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import pickle
import os

src_path = os.environ['PWD']+'/'

rnn_model_3 = {'model': model, 'tokenizer':tokenizer}
with open(src_path+'rnn_model_3.pickle', 'wb') as f:
    pickle.dump(rnn_model_3, f, pickle.HIGHEST_PROTOCOL)
# the_path = os.environ['PATH']

s3 = boto3.client('s3')
with open(src_path+"rnn_model_3.pickle", "rb") as f:
    s3.upload_fileobj(f, "recommender-project-bucket", "rnn_model_3.pickle")

#######################################
## Load model and get model predictions
#######################################

# with open('/content/drive/My Drive/g_model.pickle', 'rb') as f:
#     dese_pickles = pickle.load(f)

# model_is_back = dese_pickles['model']

# a_test = tokenizer.texts_to_sequences(X_[:5])
# a_padded_test = pad_sequences(a_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post')


# type(model_is_back)
# #hist.predict(a_padded_test)
# predictions = model_is_back.predict(x=a_padded_test)

# predictions.reshape(5)

# for prediction in predictions:
#   if prediction > .5:
#       prediction = 1
#   else:
#       prediction = 0