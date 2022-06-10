#https://petamind.com/movielens-automation-process-and-export/
#https://petamind.com/build-a-simple-recommender-system-with-matrix-factorization/
#https://petamind.com/movielens-automation-process-and-export/
import math
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.python.keras.optimizers import Adam
# import pandas as pd
from feauture_extraction.Feature_Extraction import Features_DL
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
# import keras
# from keras.layers import Input, LSTM
# from keras.layers.embeddings import Embedding
# from keras.layers.core import Dense, Dropout, Activation, Lambda
# from keras.models import Model
# from sklearn import preprocessing
# from sklearn.feature_extraction.text import CountVectorizer

from sklearn import metrics
class DataGenerator(Sequence):

    def __init__(self, dataset, vectorizer_or_sequencer, le, include_rules= "no", non_process_dataset ="", batch_size=4, dim=(1), \
                 shuffle=False, featureType ="word2vec", MAX_SEQUENCE_LENGTH = 0):
        'Initialization'
        self.dim = dim
        self.non_process_dataset = non_process_dataset
        self.include_rules = include_rules
        self.batch_size = batch_size
        self.dataset = dataset
        self.vectorizer_or_sequencer = vectorizer_or_sequencer
        self.le = le
        self.shuffle = shuffle
        self.indexes = dataset.index
        self.featureType = featureType
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.dataset) / self.batch_size)
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        idxs = [i for i in range(index*self.batch_size,(index+1)*self.batch_size)]

        # Find list of IDs
        list_IDs_temp = [self.indexes[k] for k in idxs]


        # Generate data
        fe = Features_DL()
        X = self.dataset.loc[list_IDs_temp,['tweet_text']]
        non_process_X = self.non_process_dataset.loc[list_IDs_temp,['tweet_text']]
        if self.featureType.lower() in ['tfidf', 'tf-idf', 'TfIdf', 'Tf-Idf', 'Tf-idf', 'Tfidf', "simple", 'vect', 'vec']:
            X = self.vectorizer_or_sequencer.transform(X["tweet_text"])

            X_train_Features = self.vectorizer_or_sequencer.get_feature_names()
            if self.include_rules == 'yes':

                X, X_train_Features = fe.apply_gen_rules_features_ngrams(non_process_X["tweet_text"], \
                                                                          X, X_train_Features)
            X = X.toarray()

            X = np.reshape(X, (X.shape[0], X.shape[1], 1))


        elif self.featureType.lower() in ["word2vec", "glove_word2vec", "word2vec_glove", "gloveword2vec", "word2vecglove", "google2vec", \
                                          "doc2vec", "glove" , "google", "google_news", "googlenews", "GoogleNews-vectors-negative300", "googleword2vec", "google_word2vec"]:
            X_train_Features = ["_Not_Apply_"]
            X = self.vectorizer_or_sequencer.texts_to_sequences(X["tweet_text"])

            if self.include_rules == 'yes':
                X = pad_sequences(X, maxlen=self.MAX_SEQUENCE_LENGTH - 18)
                X, _ = fe.apply_gen_rules_features_word2vec(non_process_X["tweet_text"], X, self.vectorizer_or_sequencer, X_train_Features)
                X = X.toarray()
            else:
                X = pad_sequences(X, maxlen=self.MAX_SEQUENCE_LENGTH)


        y = self.dataset.loc[list_IDs_temp,['tweet_class']]
        y = self.le.transform(y)
        y = to_categorical(y, num_classes = len(self.le.classes_), dtype ="int32")

        return X, y
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
