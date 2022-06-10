
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')#replace ignore with default for enabling the warning

import gc
import os
import csv
# import sklearn_crfsuite
# import eli5
# import xlsxwriter
import scipy
from sklearn import tree
import numpy as np

from tensorflow import keras

from tensorflow.python.keras.layers import Input, LSTM, GRU
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Lambda
# from keras.layers import LSTM, Activation, Dropout, Dense, Input, GRU, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.python.keras.layers import Dense, SimpleRNN, SpatialDropout1D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.convolutional import Convolution1D, Conv1D
from tensorflow.python.keras import backend as K
from sklearn import metrics
from utilities.my_progressBar import My_progressBar
from utilities.my_save_load_model_PipelineDL import Save_Load_Model
import os, errno
import pandas as pd
import time
import datetime
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow.keras.utils import to_categorical
class Training_Classifiers_DL:
  '''for training six classifiers and generating results.
  Note: path to the folders to save classifiers and to save results should be given individually '''
  def __init__(self, pathToResFolder = None, pathToClfFolder = None):
    '''Def: initialize pathToResFolder and PathToClfFolder Default: pathToResFolder = '../results_phase2/', pathToClfFolder = '../models_phase2/'
     Args: pass path to the folder to store results and path to folder to store classfier'''

    self.pathToClfFolder = pathToClfFolder
    self.pathToResFolder = pathToResFolder

  def show_message(self):
    print("show message training classifier irfan")

  def dictToList(self, all_metrics):
    '''
    convert dictionary to lists
    :param all_metrics:
    :return: list
    '''
    temp = [all_metrics['accuracy']]
    for item in all_metrics['precision']:
      temp.append(item)
    for item in all_metrics['recall']:
      temp.append(item)
    for item in all_metrics['f1-measure']:
      temp.append(item)
    return temp

  def limitDecimal(self, combined):
    '''round the number to two decimal places'''
    newlist = ["{0:.2f}".format(n) for n in combined]
    return newlist

  def cal_accuracy(self, y_test, y_pred_class, clf_name):
    # conf_metrics = metrics.confusion_matrix(y_test, y_pred_class)

    acc = metrics.accuracy_score(y_test, y_pred_class) * 100
    acc = "{0:.2f}".format(acc)
    # prec = metrics.precision_score(y_test, y_pred_class, average = None) * 100
    # recal = metrics.recall_score(y_test, y_pred_class, average = None) * 100
    # f1 = metrics.f1_score(y_test, y_pred_class, average = None) * 100
    # sup = metrics.sup(y_test, y_pred_class, average=None)
    # acc = metrics.accuracy_score(y_test, y_pred_class)
    prec, recal, f1, sup = metrics.precision_recall_fscore_support(y_test, y_pred_class, average = None)
    prec = prec * 100
    recal = recal * 100
    f1 = f1 * 100
    metrics_None = [prec, recal , f1]

    prec, recal, f1, sup = metrics.precision_recall_fscore_support(y_test, y_pred_class, average = 'micro')
    prec = prec * 100
    recal = recal * 100
    f1 = f1 * 100
    metrics_Micro = [prec, recal , f1]

    prec, recal, f1, sup = metrics.precision_recall_fscore_support(y_test, y_pred_class, average = 'macro')
    prec = prec * 100
    recal = recal * 100
    f1 = f1 * 100
    metrics_Macro = [prec, recal , f1]

    prec, recal, f1, sup = metrics.precision_recall_fscore_support(y_test, y_pred_class, average = 'weighted')
    prec = prec * 100
    recal = recal * 100
    f1 = f1 * 100
    metrics_Weighted = [prec, recal , f1]
    '''
    list containing set of precisions of all types (note: None gives number of precisions equals to number of
    classes, others give just single number)
    '''
    comb_pre = []
    comb_pre.extend(metrics_None[0])

    comb_pre.append(metrics_Micro[0])
    comb_pre.append(metrics_Macro[0])
    comb_pre.append(metrics_Weighted[0])
    #format of the following comb_* variable is [metric_None, metric_Micro, metric_Macro, metric_Weighted]
    #rounding decimal numbers to two decimal places
    comb_pre = self.limitDecimal(comb_pre)
    # print(comb_pre)

    '''
      list containing set of recalls of all classifiers (note: None gives number of recalls equals to number of
      classes, others give just single number)
      '''
    comb_rec = []
    comb_rec.extend(metrics_None[1])
    comb_rec.append(metrics_Micro[1])
    comb_rec.append(metrics_Macro[1])
    comb_rec.append(metrics_Weighted[1])
    #format of the following comb_* variable is [metric_None, metric_Micro, metric_Macro, metric_Weighted]
    # rounding decimal numbers to two decimal places

    comb_rec = self.limitDecimal(comb_rec)
    # print(comb_rec)


    '''
      list containing set of f1 of all types (note: None gives number of f1 equals to number of
      classes, others give just single number)
      '''
    comb_f1 = []
    comb_f1.extend(metrics_None[2])
    comb_f1.append(metrics_Micro[2])
    comb_f1.append(metrics_Macro[2])
    comb_f1.append(metrics_Weighted[2])
    # rounding decimal numbers to two decimal places
    #format of the following comb_* variable is [metric_None, metric_Micro, metric_Macro, metric_Weighted]
    comb_f1 = self.limitDecimal(comb_f1)
    all_metrics = {'accuracy': acc, 'precision': comb_pre, 'recall': comb_rec, 'f1-measure': comb_f1 }
    all_metrics_list = self.dictToList(all_metrics)
    return all_metrics_list




  #
  # def apply_GRU_KF(self, X_train_dtm, nb_classes):
  #   # print("X_train_dtm.shape", X_train_dtm.shape)
  #   X_indices = Input((X_train_dtm,))
  #   #     embedding_layer = Embedding(input_shape, 128)
  #   embedding_layer = Embedding(input_dim=X_train_dtm, \
  #                               output_dim=X_train_dtm, input_length=X_train_dtm, trainable=False)
  #   embeddings = embedding_layer(X_indices)
  #
  #   X = GRU(128, return_sequences=True)(embeddings)
  #
  #   X = Dropout(0.001)(X)
  #
  #   X = GRU(128, return_sequences=True)(X)
  #
  #   X = Dropout(0.001)(X)
  #
  #   X = GRU(128)(X)
  #   X = Dense(nb_classes, activation='softmax')(X)
  #   # if nb_classes in [1, 2]:
  #   #   X = Dense(nb_classes, activation='sigmoid')(X)
  #   # else:
  #   #   X = Dense(nb_classes, activation='softmax')(X)
  #   model = Model(inputs=X_indices, outputs=X)
  #   # model.summary()
  #   adam = keras.optimizers.Adam(learning_rate=0.0001)
  #   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
  #   # if nb_classes in [1, 2]:
  #   #   model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
  #   # else:
  #   #   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
  #   return model
  #
  # def apply_CNN_KF(self, X_train_dtm, nb_classes):
  #   nb_filter = 250
  #   filter_length = 3
  #   hidden_dims = 250
  #   nb_epoch = 2
  #
  #   def max_1d(X):
  #     return K.max(X, axis=1)
  #
  #   X_indices = Input(shape=(X_train_dtm,))
  #
  #   embedding_layer = Embedding(input_dim=X_train_dtm, \
  #                               output_dim=X_train_dtm, input_length=X_train_dtm, trainable=False)
  #   embeddings = embedding_layer(X_indices)
  #
  #   X = Convolution1D(nb_filter=nb_filter,
  #                     filter_length=filter_length,
  #                     border_mode='valid',
  #                     activation='relu',
  #                     subsample_length=1)(embeddings)
  #   X = Lambda(max_1d, output_shape=(nb_filter,))(X)
  #
  #   X = Dense(hidden_dims)(X)
  #
  #   X = Dropout(0.001)(X)
  #
  #   X = Activation('relu')(X)
  #   X = Dense(nb_classes, activation='softmax')(X)
  #   # if nb_classes in [1, 2]:
  #   #   X = Dense(nb_classes, activation='sigmoid')(X)
  #   # else:
  #   #   X = Dense(nb_classes, activation='softmax')(X)
  #
  #   model = Model(inputs=X_indices, outputs=X)
  #   adam = keras.optimizers.Adam(learning_rate=0.0001)
  #   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
  #   # if nb_classes in [1, 2]:
  #   #   model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
  #   # else:
  #   #   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
  #   return model
  #
  #
  #
  # def apply_LSTM_KF(self, X_train_dtm, nb_classes):
  #
  #   X_indices = Input(shape=(X_train_dtm,))
  #   embedding_layer = Embedding(input_dim=X_train_dtm , \
  #                               output_dim=X_train_dtm, input_length=X_train_dtm, trainable=False)
  #
  #   embeddings = embedding_layer(X_indices)
  #
  #   X = LSTM(128, return_sequences=True)(embeddings)
  #
  #   X = Dropout(0.001)(X)
  #
  #   X = LSTM(128, return_sequences=True)(X)
  #
  #   X = Dropout(0.001)(X)
  #
  #   X = LSTM(128)(X)
  #   X = Dense(nb_classes, activation='softmax')(X)
  #   # if nb_classes in [1, 2]:
  #   #   X = Dense(nb_classes, activation='sigmoid')(X)
  #   # else:
  #   #   X = Dense(nb_classes, activation='softmax')(X)
  #
  #   model = Model(inputs=X_indices, outputs=X)
  #
  #   adam = keras.optimizers.Adam(learning_rate=0.0001)
  #   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
  #   # if nb_classes in [1, 2]:
  #   #   model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
  #   # else:
  #   #   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
  #   # model.summary()
  #   return model

  # def apply_Seq_Test(self, X_train_dtm, nb_classes):
  #   from keras.models import Sequential
  #   model = Sequential()
  #   model.add(Dense(units=X_train_dtm, input_shape=(X_train_dtm,1) ))
  #   model.add(Dense(units=X_train_dtm, activation='relu'))
  #   model.add(Dense(nb_classes, activation='softmax'))
  #   # compile the keras model
  #   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  #   return model

  def apply_RNN_KF_ngrams(self, X_train_dtm, nb_classes):

    X_indices = Input(shape=(X_train_dtm,1))
    X_train_dtm = int(X_train_dtm / 10)
    X = SimpleRNN( X_train_dtm, return_sequences=False)(X_indices)
    # X = Dense(units=X_train_dtm)(X)
    # X = Dropout(0.1)(X)
    X = Dense(units=X_train_dtm, activation='relu')(X)
    X = Dense(nb_classes, activation='softmax')(X)
    model = Model(inputs=X_indices, outputs=X)
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    # if nb_classes in [1, 2]:
    #   model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # else:
    #   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # print("model")
    # print(model.summary())
    return model


  def apply_GRU_KF_ngrams(self, X_train_dtm, nb_classes):

    X_indices = Input(shape=(X_train_dtm,1))

    X_train_dtm = int(X_train_dtm / 10)
    X = GRU(X_train_dtm, return_sequences=False)(X_indices)

    # X = Dense(units=X_train_dtm)(X)
    # X = Dropout(0.1)(X)
    X = Dense(units=X_train_dtm, activation='relu')(X)
    X = Dense(nb_classes, activation='softmax')(X)
    model = Model(inputs=X_indices, outputs=X)
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    # if nb_classes in [1, 2]:
    #   model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # else:
    #   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


  def apply_LSTM_KF_ngrams(self, X_train_dtm, nb_classes):

    X_indices = Input(shape=(X_train_dtm,1))
    X_train_dtm = int(X_train_dtm/10)
    X = LSTM(X_train_dtm, return_sequences=False)(X_indices)
    # X = Dense(units=X_train_dtm)(X)
    # X = Dropout(0.1)(X)
    X = Dense(units=X_train_dtm, activation='relu')(X)
    X = Dense(nb_classes, activation='softmax')(X)
    model = Model(inputs=X_indices, outputs=X)
    adam = keras.optimizers.Adam(learning_rate=0.01)
    # if nb_classes in [1, 2]:
    #   model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # else:
    #   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

  def apply_CNN_KF_ngrams(self, X_train_dtm, nb_classes):
    filter = 250
    kernel_size = 4
    hidden_dims = 250

    def max_1d(X):
      return K.max(X, axis=1)

    X_indices = Input(shape=(X_train_dtm,1))



    X = Conv1D(
      filters=filter,
      kernel_size= kernel_size,
      padding='same',
      activation='relu')(X_indices)
    X = Lambda(max_1d, output_shape=(filter,))(X)

    X = Dense(hidden_dims)(X)

    X = Dropout(0.001)(X)

    X = Activation('relu')(X)

    # if nb_classes in [1, 2]:
    #   X = Dense(nb_classes, activation='sigmoid')(X)
    # else:
    #   X = Dense(nb_classes, activation='softmax')(X)
    X = Dense(nb_classes, activation='softmax')(X)

    model = Model(inputs=X_indices, outputs=X)
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    # if nb_classes in [1, 2]:
    #   model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # else:
    #   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

  def apply_RNN_KF_DL(self, nb_classes, vocab_len, \
                      embed_vector_len, input_shape, emb_matrix):
    # print("X_train_dtm.shape", X_train_dtm.shape)
    X_indices = Input(shape=(input_shape,), dtype='int32')
    #     embedding_layer = Embedding(input_shape, 128)
    embedding_layer = Embedding(input_dim=vocab_len, \
                                output_dim=embed_vector_len, input_length=input_shape, weights=[emb_matrix],
                                trainable=False)
    embeddings = embedding_layer(X_indices)

    X = SimpleRNN(128, return_sequences=True)(embeddings)

    X = Dropout(0.001)(X)

    X = SimpleRNN(128, return_sequences=True)(X)

    X = Dropout(0.001)(X)

    X = SimpleRNN(128)(X)

    # if nb_classes in [1, 2]:
    #   X = Dense(nb_classes, activation='sigmoid')(X)
    # else:
    #   X = Dense(nb_classes, activation='softmax')(X)
    X = Dense(nb_classes, activation='softmax')(X)

    model = Model(inputs=X_indices, outputs=X)
    # model.summary()
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # if nb_classes in [1, 2]:
    #   model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # else:
    #   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    return model

  def apply_GRU_KF_DL(self, nb_classes, vocab_len, \
                      embed_vector_len, input_shape, emb_matrix):
    # print("X_train_dtm.shape", X_train_dtm.shape)
    X_indices = Input((input_shape,))
    #     embedding_layer = Embedding(input_shape, 128)
    embedding_layer = Embedding(input_dim=vocab_len , \
                                output_dim=embed_vector_len, input_length=input_shape, weights=[emb_matrix],
                                trainable=False)
    embeddings = embedding_layer(X_indices)

    X = GRU(128, return_sequences=True)(embeddings)

    X = Dropout(0.001)(X)

    X = GRU(128, return_sequences=True)(X)

    X = Dropout(0.001)(X)

    X = GRU(128)(X)

    # if nb_classes in [1, 2]:
    #   X = Dense(nb_classes, activation='sigmoid')(X)
    # else:
    #   X = Dense(nb_classes, activation='softmax')(X)
    X = Dense(nb_classes, activation='softmax')(X)

    model = Model(inputs=X_indices, outputs=X)
    # model.summary()
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # if nb_classes in [1, 2]:
    #   model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # else:
    #   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

  def apply_LSTM_KF_DL(self, nb_classes, vocab_len, \
                       embed_vector_len, input_shape, emb_matrix):

    X_indices = Input(shape=(input_shape,), dtype='int32')
    embedding_layer = Embedding(input_dim=vocab_len , \
                                output_dim=embed_vector_len, input_length=input_shape, weights=[emb_matrix], trainable=False)

    embeddings = embedding_layer(X_indices)

    X = LSTM(128, return_sequences=True)(embeddings)

    X = Dropout(0.001)(X)

    X = LSTM(128, return_sequences=True)(X)

    X = Dropout(0.001)(X)

    X = LSTM(128)(X)

    # if nb_classes in [1, 2]:
    #   X = Dense(nb_classes, activation='sigmoid')(X)
    # else:
    #   X = Dense(nb_classes, activation='softmax')(X)
    X = Dense(nb_classes, activation='softmax')(X)

    model = Model(inputs=X_indices, outputs=X)

    adam = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # if nb_classes in [1, 2]:
    #   model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # else:
    #   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

    filter = 250
    kernel_size = 3
    hidden_dims = 250

    def max_1d(X):
      return K.max(X, axis=1)

    X_indices = Input(shape=(X_train_dtm,1))



    X = Conv1D(
      filters=filter,
      kernel_size= kernel_size,
      padding='same',
      activation='relu')(X_indices)
    X = Lambda(max_1d, output_shape=(filter,))(X)

  def apply_CNN_KF_DL(self, nb_classes, vocab_len, \
                      embed_vector_len, input_shape, emb_matrix):
    filter = 250
    kernel_size = 3
    hidden_dims = 250

    def max_1d(X):
      return K.max(X, axis=1)

    X_indices = Input(shape=(input_shape,), dtype='int32')

    embedding_layer = Embedding(input_dim=vocab_len , \
                                output_dim=embed_vector_len, input_length=input_shape, weights=[emb_matrix], trainable=False)
    embeddings = embedding_layer(X_indices)

    X = Conv1D(
      filters=filter,
      kernel_size= kernel_size,
      padding='same',
      activation='relu')(embeddings)
    X = Lambda(max_1d, output_shape=(filter,))(X)

    X = Dense(hidden_dims)(X)

    X = Dropout(0.001)(X)

    X = Activation('relu')(X)

    # if nb_classes in [1, 2]:
    #   X = Dense(nb_classes, activation='sigmoid')(X)
    # else:
    #   X = Dense(nb_classes, activation='softmax')(X)
    X = Dense(nb_classes, activation='softmax')(X)

    model = Model(inputs=X_indices, outputs=X)
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # if nb_classes in [1, 2]:
    #   model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # else:
    #   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

  def ls_ToDf(self ,ls1 , ls2, ls3, ls4, ls5, ls6):
    '''
    -convert list type into DataFrame and add columns names for creating labelled table of dataframe type
    + takes six lists of arguments and convert it Dataframe row wise order
    '''
    ls_digit1 = list(map(float, ls1))
    ls_digit2 = list(map(float, ls2))
    ls_digit3 = list(map(float, ls3))
    ls_digit4 = list(map(float, ls4))
    ls_digit5 = list(map(float, ls5))
    ls_digit6 = list(map(float, ls6))
    res = list()
    res.append(ls_digit1)
    res.append(ls_digit2)
    res.append(ls_digit3)
    res.append(ls_digit4)
    res.append(ls_digit5)
    res.append(ls_digit6)
    df = pd.DataFrame(res, columns=['Accuracy', 'Precision', 'Recall', 'F1-Measure'])
    df['Classifier']=['Naive Bayes', 'Logisitic Regression', 'SVM', 'Random Forest', 'Gradient Boosting', 'NLP']
    df = df[['Classifier','Accuracy', 'Precision', 'Recall', 'F1-Measure']]
    df.set_index('Classifier', inplace = True)
    return df


  def saving_Clf_toDisk(self, f_name, num_Clf, save_clf, clf_nb, clf_logreg, clf_randomForest, clf_gb, clf_svm, clf_MLP, clf_DT):


    phase_label = ""
    f_name =  "_" + f_name
    message = 'Saving Classifiers to disk for ' + phase_label
    pBar = My_progressBar(message, num_Clf)
    start_time = time.time()
    save_clf.save_Model(clf_nb, f_name + '_NB' + phase_label, 'NB')
    pBar.call_to_progress(start_time)
    start_time = time.time()
    save_clf.save_Model(clf_logreg, f_name + '_LR'+ phase_label, "LR")
    pBar.call_to_progress(start_time)
    start_time = time.time()
    save_clf.save_Model(clf_svm, f_name + '_SVM'+ phase_label, "SVM")
    pBar.call_to_progress(start_time)
    start_time = time.time()
    save_clf.save_Model(clf_gb, f_name + '_GB'+ phase_label, 'GB')
    pBar.call_to_progress(start_time)
    start_time = time.time()
    save_clf.save_Model(clf_randomForest, f_name + '_RF'+ phase_label, 'RF')
    pBar.call_to_progress(start_time)
    start_time = time.time()
    save_clf.save_Model(clf_MLP, f_name + '_MLP'+ phase_label, 'MLP')
    pBar.call_to_progress(start_time)
    save_clf.save_Model(clf_DT, f_name + '_DT'+ phase_label, 'DT')
    pBar.call_to_progress(start_time)



  def save_ResultsCSV(self, filename, results_list, rName):
    with open(filename, 'a') as fh:
      tw = csv.writer(fh)
      tw.writerow(results_list)
      print(rName + "'s Results saved successfully.")

  def delete_rows_csr(self, mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
      raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

  def list_str(self, list1):
    '''Def: convert list to string by joing list element using _
    Args: pass python list as an argument
    Ret: retrun resultant string'''
    return '_'.join(list1)
# (nb_classes, vocab_len, embed_vector_len, MAX_SEQUENCE_LENGTH, emb_matrix)



  def check_file(self, le_name_mapping, fname, path = ''):#path is pathToFolder e.g., '../model_phase2/'
    if path == '':
      print('please provide path to the folder:')
      exit(0)


    path = path + fname + '.csv'
    with open(path, 'wt') as fh:
      tw = csv.writer(fh)
      header = ['Classifiers','Features','Accuracy']
      header.extend(le_name_mapping)
      header.extend(['Micro_Precision', 'Macro_Precision', 'Weighted_precision'])
      header.extend(le_name_mapping)
      header.extend(['Micro_Recall', 'Macro_Recall', 'Weighted_Recall'])
      header.extend(le_name_mapping)
      header.extend(['Micro_F1-Measure', 'Macro_F1-Measure', 'Weighted_F1-Measure'])
      tw.writerow(header)
    # while(os.path.isfile(path) == True):
    #   old = path
    #   temp +=' copy'
    #   path =temp +'.xlsx'
    # workbook = xlsxwriter.Workbook(path)
    # workbook.close()
    return path

  def extractInfo_specs(self, st, req):
    '''Def: fetch required info from list
    Args: pass list and req arguments where req contained name of the required info to be fetched from list
    Ret: return required info'''
    st = st.split('_')
    if req == 'sheet_name':
      info = st[2]
    elif req == 'file_name':
      info = self.list_str(st[0:2]) + '_' + self.list_str(st[3:])
    elif req == 'freq':
      info = st[-1].split('-')[1]
    return info

# (train_generator, dataset["tweet_class"], epochs, vect_sequencer, le, pathTofileName, f_name, \
#  vocab_len, fun_word_to_vec_map, embed_vector_len, emb_matrix, include_rules)

  def initiate_clf_training_word2vec(self, train_generator, y_org, epochs, le, pathToResultFilename, f_name, \
                                     MAX_SEQUENCE_LENGTH, vocab_len, embed_vector_len, emb_matrix):
    from sklearn.metrics import accuracy_score
    from sklearn import preprocessing
    # if include_rules:
    #   size = len(vect_sequencer.get_feature_names()) + 18
    # else:
    #   size = len(vect_sequencer.get_feature_names())
    nb_classes = len(le.classes_)
    y = le.transform(y_org)
    feature = '_'.join(f_name.split('_')[2:])

    # RNN
    clf_label = 'RNN'
    model = self.apply_RNN_KF_DL(nb_classes, vocab_len, embed_vector_len, MAX_SEQUENCE_LENGTH, emb_matrix)
    history = model.fit_generator(train_generator, epochs=epochs)
    y_pred = model.predict_generator(train_generator)
    y_pred = np.argmax(y_pred, axis=1)

    score = self.cal_accuracy(y, y_pred, clf_label)
    score.insert(0, clf_label)
    score.insert(1, feature)
    self.save_ResultsCSV(pathToResultFilename, score, clf_label)

    # GRU
    clf_label = 'GRU'
    model = self.apply_GRU_KF_DL(nb_classes, vocab_len, embed_vector_len, MAX_SEQUENCE_LENGTH, emb_matrix)
    history = model.fit_generator(train_generator, epochs=epochs)
    y_pred = model.predict_generator(train_generator)
    y_pred = np.argmax(y_pred, axis=1)

    score = self.cal_accuracy(y, y_pred, clf_label)
    score.insert(0, clf_label)
    score.insert(1, feature)
    self.save_ResultsCSV(pathToResultFilename, score, clf_label)

    # LSTM
    clf_label = 'LSTM'
    model = self.apply_LSTM_KF_DL(nb_classes, vocab_len, embed_vector_len, MAX_SEQUENCE_LENGTH, emb_matrix)
    history = model.fit_generator(train_generator, epochs=epochs)
    y_pred = model.predict_generator(train_generator)
    y_pred = np.argmax(y_pred, axis=1)

    score = self.cal_accuracy(y, y_pred, clf_label)
    score.insert(0, clf_label)
    score.insert(1, feature)
    self.save_ResultsCSV(pathToResultFilename, score, clf_label)

    # CNN
    clf_label = 'CNN'
    model = self.apply_CNN_KF_DL(nb_classes, vocab_len, embed_vector_len, MAX_SEQUENCE_LENGTH, emb_matrix)
    history = model.fit_generator(train_generator, epochs=epochs)
    y_pred = model.predict_generator(train_generator)
    y_pred = np.argmax(y_pred, axis=1)

    score = self.cal_accuracy(y, y_pred, clf_label)
    score.insert(0, clf_label)
    score.insert(1, feature)
    self.save_ResultsCSV(pathToResultFilename, score, clf_label)


  def initiate_clf_training_ngrams(self, train_generator, batch_size, y_org, epochs, vect, le, pathToResultFilename, f_name, \
                                   include_rules = False):

    if include_rules == "yes":
      size = len(vect.get_feature_names()) + 18
    else:
      size = len(vect.get_feature_names())

    nb_classes = len(le.classes_)
    y = le.transform(y_org)
    feature = '_'.join(f_name.split('_')[2:])

    clf_label = 'RNN'

    model = self.apply_RNN_KF_ngrams(size, nb_classes)
    # print(model.summary())
    history = model.fit_generator(train_generator, epochs=epochs)

    y_pred = model.predict_generator(train_generator)

    y_pred = np.argmax(y_pred, axis=1)

    score = self.cal_accuracy(y, y_pred, clf_label)
    score.insert(0, clf_label)
    score.insert(1, feature)
    self.save_ResultsCSV(pathToResultFilename, score, clf_label)




    # GRU
    clf_label = 'GRU'
    model = self.apply_GRU_KF_ngrams(size, nb_classes)
    history = model.fit_generator(train_generator, epochs=epochs)
    y_pred = model.predict_generator(train_generator)
    y_pred = np.argmax(y_pred, axis=1)

    score = self.cal_accuracy(y, y_pred, clf_label)
    score.insert(0, clf_label)
    score.insert(1, feature)
    self.save_ResultsCSV(pathToResultFilename, score, clf_label)

    # LSTM
    clf_label = 'LSTM'
    model = self.apply_LSTM_KF_ngrams(size, nb_classes)
    # print(model.summary())
    history = model.fit_generator(train_generator, epochs=epochs)
    y_pred = model.predict_generator(train_generator)
    y_pred = np.argmax(y_pred, axis=1)

    score = self.cal_accuracy(y, y_pred, clf_label)
    score.insert(0, clf_label)
    score.insert(1, feature)
    self.save_ResultsCSV(pathToResultFilename, score, clf_label)

    # CNN
    clf_label = 'CNN'
    model = self.apply_CNN_KF_ngrams(size, nb_classes)
    print(model.summary())
    history = model.fit_generator(train_generator, epochs=epochs)
    y_pred = model.predict_generator(train_generator)
    y_pred = np.argmax(y_pred, axis=1)

    score = self.cal_accuracy(y, y_pred, clf_label)
    score.insert(0, clf_label)
    score.insert(1, feature)
    self.save_ResultsCSV(pathToResultFilename, score, clf_label)

  def create_Folder(self, file_path):
    '''Def: To create folder
    Args: pass folder path with full name to create it
    Ret: null'''
    try:
      os.makedirs(file_path)
      print('Folder, \" ' + file_path + "\" is created successfully.")
    except OSError as e:
      print('Folder, \" ' + file_path + "\" might already exists.")
      if e.errno != errno.EEXIST:
        raise
  # train_generator, epochs, vect, le, dataset, uniLabel, featureTypeFolderLabel

  def training_clfs_ngrams(self, train_generator, batch_size, epochs, vect, le, dataset, f_name, featureType, \
                           include_rules = False):


    if ((featureType == None)):
      print("provide feature type")
      exit(0)
    pathToResFolder = '../' + featureType + '_results/'
    self.create_Folder(pathToResFolder)

    pathTofileName = self.extractInfo_specs(f_name, 'file_name')
    pathTofileName = self.check_file(le.classes_, pathTofileName, pathToResFolder)
    self.initiate_clf_training_ngrams(train_generator, batch_size, dataset["tweet_class"], \
                                      epochs, vect, le, pathTofileName, f_name, include_rules)


# (train_generator, epochs, vect_sequencer, le, dataset, label, vocab_len, fun_word_to_vec_map, embed_vector_len, emb_matrix, \
#  featureType = vect_type, include_rules=include_rules)
  def training_clfs_word2vec(self, train_generator, epochs, le, dataset, f_name, MAX_SEQUENCE_LENGTH, vocab_len, embed_vector_len, emb_matrix, \
                             featureType):


    if ((featureType == None)):
      print("provide feature type")
      exit(0)
    pathToResFolder = '../' + featureType + '_results/'
    self.create_Folder(pathToResFolder)


    pathTofileName = self.extractInfo_specs(f_name, 'file_name')
    pathTofileName = self.check_file(le.classes_, pathTofileName, pathToResFolder)
    # (self, train_generator,y_org, epochs, vect, le, dataset, pathToResultFilename, f_name)
    self.initiate_clf_training_word2vec(train_generator, dataset["tweet_class"], epochs, le, pathTofileName, f_name, \
                                        MAX_SEQUENCE_LENGTH, vocab_len, embed_vector_len, emb_matrix)








