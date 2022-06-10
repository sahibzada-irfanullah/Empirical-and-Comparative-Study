import time
import warnings
warnings.filterwarnings('ignore')#replace ignore with default for enabling the warning
from sklearn import preprocessing
import pandas as pd
from feauture_extraction.Feature_Extraction import Features_DL
# from training_clf.Training_classifier_Pipeline_ContextFeatML import Training_Classifiers
import datetime
import os, errno
import numpy as np
def create_Folder(file_path):
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
def initiateTraining(filename, feature_type_folderLabel, include_rules, epochs, embed_dim, batch_size = 10):
    # fnameTrainingDataset = "/home/sahibzada/ipynbTests/dual labels/specific_monImb_OtherImb_pre_dual.csv"
    path_to_preprocessedDataset = "../datasets/preprocessed_" + filename
    path_to_nonPreprocessedDataset = "../datasets/" + filename

    dataset = pd.read_csv(path_to_preprocessedDataset)

    # X = df['tweet_text']
    # batch_size = 10

    # le = le.fit(dataset.iloc[:, 8])




    #generating dictionary containing label encoder mapping for each class
    # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # print(le_name_mapping)

    # feature_type_folderLabel = "word2vec"

    feat_extractWor2Vec = Features_DL()

    feat_extractWor2Vec.generate_Features(dataset, path_to_nonPreprocessedDataset, \
                                                           feature_type_folderLabel, epochs = epochs, batch_size= batch_size, embed_dim = embed_dim, minDF=1, maxDF=1.0, \
                                                           include_rules= include_rules)#, featureType=feature_type_folderLabel)
    # check X_train_dtm, indices_uniq = np.unique(X_train_dtm, axis=0) line
    # in word2vec_Vect
    # for k, v in features_encap.items():
    #     print("key", k, "value", v['specs'])
    # tc_encap = Training_Classifiers()
    #
    # tc_encap.training_clfs(features_encap, classes_labels,
    #                        k_fold = k_fold, featureTypeFolderLabel=feature_type_folderLabel)

if __name__ == '__main__':
    script_start = time.time()  # script start point
    filename = "500_random_sample.csv"
    epochs = 1
    embed_dim = 300
    batch_size = 8
    feature_type_folderLabel = "simple"
    include_rules = "no"

    initiateTraining(filename, feature_type_folderLabel, include_rules, epochs, embed_dim, batch_size)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # feature_type_folderLabel = "simple"
    # include_rules = "no"
    #
    # initiateTraining(filename, feature_type_folderLabel, include_rules, k_fold, embed_dim)
    # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # feature_type_folderLabel = "tfidf"
    # include_rules = "no"
    #
    # initiateTraining(filename, feature_type_folderLabel, include_rules, k_fold, embed_dim)
    # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # feature_type_folderLabel = "tfidf"
    # include_rules = "yes"
    #
    # initiateTraining(filename, feature_type_folderLabel, include_rules, k_fold, embed_dim)
    # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # feature_type_folderLabel = "word2vec"
    # include_rules = "no"
    #
    # initiateTraining(filename, feature_type_folderLabel, include_rules, k_fold, embed_dim)
    # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # feature_type_folderLabel = "word2vec"
    # include_rules = "yes"
    #
    # initiateTraining(filename, feature_type_folderLabel, include_rules, k_fold, embed_dim)
    # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


