import time
import warnings
warnings.filterwarnings('ignore')#replace ignore with default for enabling the warning
from sklearn import preprocessing
import pandas as pd
from feauture_extraction.Feature_Extraction_ML import Features_ML
from training_clf.Training_classifiers_ML import Training_Classifiers_ML
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

def initiateTraining_ML(filename, feature_type_folderLabel, include_rules, k_fold, embed_dim):
    # fnameTrainingDataset = "/home/sahibzada/ipynbTests/dual labels/specific_monImb_OtherImb_pre_dual.csv"
    path_to_preprocessedDataset = "../datasets/preprocessed_" + filename
    path_to_nonPreprocessedDataset = "../datasets/" + filename

    df = pd.read_csv(path_to_preprocessedDataset)

    # X = df['tweet_text']
    y_binary = df['tweet_class']

    le = preprocessing.LabelEncoder()
    y_binary = le.fit_transform(y_binary)

    classes_labels_binary = le.classes_
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))


    #generating dictionary containing label encoder mapping for each class
    # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # print(le_name_mapping)

    # feature_type_folderLabel = "word2vec"

    feat_extractWor2Vec = Features_ML()

    features_encap = feat_extractWor2Vec.generate_Features(df, y_binary, path_to_nonPreprocessedDataset, feature_type_folderLabel, path2word2vecModel ="", \
                                                           embed_dim = embed_dim, minDF=1, maxDF=1.0, include_rules= include_rules, featureTypeFolderLabel=feature_type_folderLabel)
    # duplicates removed after removing duplicates from sequences check .
    # check X_train_dtm, indices_uniq = np.unique(X_train_dtm, axis=0) line
    # in word2vec_Vect
    # for k, v in features_encap.items():
    #     print("key", k, "value", v['specs'])
    tc_encap = Training_Classifiers_ML()

    tc_encap.training_clfs(features_encap, classes_labels_binary,
                           k_fold = k_fold, featureTypeFolderLabel=feature_type_folderLabel)

if __name__ == '__main__':
    script_start = time.time()  # script start point
    filename = "binary_dataset_Copy.csv"
    k_fold = 10
    embed_dim = 300
    feature_type_folderLabel = "simple"
    include_rules = "yes"

    initiateTraining_ML(filename, feature_type_folderLabel, include_rules, k_fold, embed_dim)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    feature_type_folderLabel = "simple"
    include_rules = "no"

    initiateTraining_ML(filename, feature_type_folderLabel, include_rules, k_fold, embed_dim)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    feature_type_folderLabel = "tfidf"
    include_rules = "no"

    initiateTraining_ML(filename, feature_type_folderLabel, include_rules, k_fold, embed_dim)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    feature_type_folderLabel = "tfidf"
    include_rules = "yes"

    initiateTraining_ML(filename, feature_type_folderLabel, include_rules, k_fold, embed_dim)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    feature_type_folderLabel = "word2vec"
    include_rules = "no"

    initiateTraining_ML(filename, feature_type_folderLabel, include_rules, k_fold, embed_dim)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    feature_type_folderLabel = "word2vec"
    include_rules = "yes"

    initiateTraining_ML(filename, feature_type_folderLabel, include_rules, k_fold, embed_dim)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print("\nMain script completed")
    Total_time = time.time() - script_start
    print("\nTotal time for script completion :" + str(datetime.timedelta(seconds=int(Total_time))))

