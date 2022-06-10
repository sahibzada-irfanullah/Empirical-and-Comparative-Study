# coding: utf-8

# In[44]:
import warnings

warnings.filterwarnings('ignore')  # replace ignore with default for enabling the warning
# import sys
# sys.path.append("/home/sahibzada/ipynb files/request_identification")
import pandas as pd
from utilities.my_progressBar import My_progressBar
from utilities.my_save_load_model_ML import Save_Load_Model
import scipy
import time
import numpy as np
import gensim
from tqdm import tqdm
import datetime
# from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from tensorflow.python.keras.preprocessing.text import Tokenizer
from utilities.my_save_load_model_ML import Save_Load_Model
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from gensim.models.doc2vec import TaggedDocument
import os, errno

class Features_ML:
    '''Generating Ngrams and rules features'''

    def apply_regex_ngrams(self, text, regex):
        match_found = (re.search(regex, text) != None)
        match_found = int(match_found == True)
        return match_found

    def apply_regex_word2vec(self, text, regex, ruleTokenizerSequenceIndex):
        match_found = (re.search(regex, text) != None)
        if match_found:
            match_found = int(ruleTokenizerSequenceIndex)
        else:
            match_found = 0
        return match_found

    def gen_rules_features_ngrams(self, X_data_series):  # , X_data_dtm, features_arg):
        '''sparse matrix and series matrices should be converted to dataframe for applying rules and treating
        it as features...
        I wrote two functions i.e.,sparse_matrix_to_DataFrame() and series_DataFrame()
          for changing datatypes'''
        X_data_DF = self.series_to_DataFrame(X_data_series)
        regexes = [
            re.compile(r'\b(I|we)\b.*\b(am|are|will be)\b.*\b(bringing|giving|helping|raising|donating|auctioning)\b',
                       re.I | re.M),
            re.compile(r'\b(I\'m)\b.*\b(bringing|giving|helping|raising|donating|auctioning)\b', re.I | re.M),
            re.compile(r'\b(we\'re)\b.*\b(bringing|giving|helping|raising|donating|auctioning)\b', re.I | re.M),
            re.compile(r'\b(I|we)\b.*\b(will|would like to)\b.*\b(bring|give|help|raise|donate|auction)\b',
                       re.I | re.M),
            re.compile(r'\b(I|we)\b.*\b(will|would like to)\b.*\b(work|volunteer|assist)\b', re.I | re.M),
            re.compile(r'\b(we\'ll)\b.*\b(bring|give|help|raise|donate|auction)\b', re.I | re.M),
            re.compile(r'\b(I|we)\b.*\b(ready|prepared)\b.*\b(bring|give|help|raise|donate|auction)\b', re.I | re.M),
            re.compile(r'\b(where)\b.*\b(can I|can we)\b.*\b(bring|give|help|raise|donate)\b', re.I | re.M),
            re.compile(r'\b(where)\b.*\b(can I|can we)\b.*\b(work|volunteer|assist)\b', re.I | re.M),
            re.compile(r'\b(I|we)\b.*\b(like|want)\b.*\bto\b.*\b(bring|give|help|raise|donate)\b', re.I | re.M),
            re.compile(r'\b(I|we)\b.*\b(like|want)\b.*\bto\b.*\b(work|volunteer|assist)\b', re.I | re.M),
            re.compile(r'\b(will be)\b.*\b(brought|given|raised|donated|auctioned)\b', re.I | re.M),
            re.compile(r'\b\w*\s*\b\?', re.I | re.M),
            re.compile(r'\b(you|u).*(can|could|should|want to)\b', re.I | re.M),
            re.compile(r'\b(can|could|should).*(you|u)\b', re.I | re.M),
            re.compile(r'\b(like|want)\b.*\bto\b.*\b(bring|give|help|raise|donate)\b', re.I | re.M),
            re.compile(r'\b(how)\b.*\b(can I|can we)\b.*\b(bring|give|help|raise|donate)\b', re.I | re.M),
            re.compile(r'\b(how)\b.*\b(can I|can we)\b.*\b(work|volunteer|assist)\b', re.I | re.M)

        ]
        temp = pd.DataFrame()
        features_arg = []
        for i, regex in zip(range(len(regexes)), regexes):
            columnName = "RegEx_" + str(i + 1)
            features_arg.append(columnName)
            temp[columnName] = X_data_DF['tweet_text'].apply(lambda text: self.apply_regex_ngrams(text, regex))
        temp_sparse = scipy.sparse.csr_matrix(temp.values)
        return temp_sparse, features_arg

    def gen_rules_features_word2vec(self, X_data_series, tokenizer):  # , X_data_dtm, features_arg):
        '''sparse matrix and series matrices should be converted to dataframe for applying rules and treating
    it as features...
    I wrote two functions i.e.,sparse_matrix_to_DataFrame() and series_DataFrame()
      for changing datatypes'''
        X_data_DF = self.series_to_DataFrame(X_data_series)
        regexes = [
            re.compile(r'\b(I|we)\b.*\b(am|are|will be)\b.*\b(bringing|giving|helping|raising|donating|auctioning)\b',
                       re.I | re.M),
            re.compile(r'\b(I\'m|Im)\b.*\b(bringing|giving|helping|raising|donating|auctioning)\b', re.I | re.M),
            re.compile(r'\b(we\'re|we are)\b.*\b(bringing|giving|helping|raising|donating|auctioning)\b', re.I | re.M),
            re.compile(r'\b(I|we)\b.*\b(will|would like to)\b.*\b(bring|give|help|raise|donate|auction)\b',
                       re.I | re.M),
            re.compile(r'\b(I|we)\b.*\b(will|would like to)\b.*\b(work|volunteer|assist)\b', re.I | re.M),
            re.compile(r'\b(we\'ll|we will)\b.*\b(bring|give|help|raise|donate|auction)\b', re.I | re.M),
            re.compile(r'\b(I|we)\b.*\b(ready|prepared)\b.*\b(bring|give|help|raise|donate|auction)\b', re.I | re.M),
            re.compile(r'\b(where)\b.*\b(can I|can we)\b.*\b(bring|give|help|raise|donate)\b', re.I | re.M),
            re.compile(r'\b(where)\b.*\b(can I|can we)\b.*\b(work|volunteer|assist)\b', re.I | re.M),
            re.compile(r'\b(I|we)\b.*\b(like|want)\b.*\bto\b.*\b(bring|give|help|raise|donate)\b', re.I | re.M),
            re.compile(r'\b(I|we)\b.*\b(like|want)\b.*\bto\b.*\b(work|volunteer|assist)\b', re.I | re.M),
            re.compile(r'\b(will be)\b.*\b(brought|given|raised|donated|auctioned)\b', re.I | re.M),
            re.compile(r'\b\w*\s*\b\?', re.I | re.M),
            re.compile(r'\b(you|u).*(can|could|should|want to)\b', re.I | re.M),
            re.compile(r'\b(can|could|should).*(you|u)\b', re.I | re.M),
            re.compile(r'\b(like|want)\b.*\bto\b.*\b(bring|give|help|raise|donate)\b', re.I | re.M),
            re.compile(r'\b(how)\b.*\b(can I|can we)\b.*\b(bring|give|help|raise|donate)\b', re.I | re.M),
            re.compile(r'\b(how)\b.*\b(can I|can we)\b.*\b(work|volunteer|assist)\b', re.I | re.M)

        ]
        temp = pd.DataFrame()
        features_arg = []
        for i, regex in zip(range(len(regexes)), regexes):
            # we can also use ruleString()
            columnName = "rule" + str(i + 1)
            features_arg.append(columnName)
            temp[columnName] = X_data_DF['tweet_text'].apply(lambda text: self.apply_regex_word2vec(text, regex, tokenizer.word_index[columnName]))

        temp_sparse = scipy.sparse.csr_matrix(temp.values)

        return temp_sparse, features_arg

    def concat_sparse_matrices_h(self, data_X_dtm, data_Rules_dtm, features_X, features_Rules):
        combined_features = features_X + features_Rules
        concat_sparse = scipy.sparse.hstack([data_X_dtm, data_Rules_dtm], format='csr')
        return concat_sparse, combined_features

    def gen_Ngrams(self, X_train, X_test, lower, higher):
        Vect = CountVectorizer(ngram_range=(lower, higher))
        X_train_dtm = Vect.fit_transform(X_train)
        X_test_dtm = Vect.transform(X_test)
        return X_train_dtm, X_test_dtm, Vect.get_feature_names()

    def sparse_matrix_to_DataFrame(self, X_data_dtm, features):
        X_data_dtm = pd.DataFrame(X_data_dtm.toarray(), columns=features)
        return X_data_dtm

    def series_to_DataFrame(self, X_data):
        X_data = X_data.to_frame()
        return X_data

    def apply_gen_rules_features_ngrams(self, X, X_train_WoR_dtm, X_train_WoR_Features):
        X_Rules_dtm, features_Rules = self.gen_rules_features_ngrams(X)
        X_train_WR_dtm, combined_features = self.concat_sparse_matrices_h(X_train_WoR_dtm, X_Rules_dtm,
                                                                          X_train_WoR_Features, features_Rules)
        return X_train_WR_dtm, combined_features
    def apply_gen_rules_features_word2vec(self, X, X_train_WoR_dtm, tokenizer, X_train_WoR_Features):
        X_Rules_dtm, features_Rules = self.gen_rules_features_word2vec(X, tokenizer)

        X_train_WR_dtm, combined_features = self.concat_sparse_matrices_h(X_train_WoR_dtm, X_Rules_dtm,
                                                                          X_train_WoR_Features, features_Rules)
        return X_train_WR_dtm, combined_features

    def simple_CountVect(self, X, path_to_nonPreprocessedDataset, lower, higher, minDF, maxDF, include_rules):
        Vect = CountVectorizer(ngram_range=(lower, higher), min_df=minDF, max_df=maxDF)
        # X_train_WoR_dtm = Vect.fit_transform(X)

        Vect.fit(X)
        X_train_WoR_dtm = Vect.transform(X)
        X_train_Features = Vect.get_feature_names()
        if include_rules == 'yes':
            df_notPreprocessed = pd.read_csv(path_to_nonPreprocessedDataset,  encoding = "ISO-8859-1")
            X_train_dtm, X_train_Features = self.apply_gen_rules_features_ngrams(df_notPreprocessed["tweet_text"], X_train_WoR_dtm, X_train_Features)
        else:
            X_train_dtm = X_train_WoR_dtm
        # removing duplicates from the sequences generated from text data
        X_train_dtm, indices_uniq = np.unique(X_train_dtm.toarray(), return_index=True, axis=0)
        return X_train_dtm, indices_uniq, Vect

    def setPathToVectFolder(self, pathToVectFolder='../vect_phase2/'):
        '''Def: set path to the folder containing vectorizers
    Args: pass path to the folder to folder containing vectorizers as an argument (Default: '../vectorizer/')
     Ret: none'''
        self.pathToVectFolder = pathToVectFolder

    def tfIdf_Vect(self, X, path_to_nonPreprocessedDataset, lower, higher, minDF, maxDF, include_rules):
        Vect = TfidfVectorizer(ngram_range=(lower, higher), min_df=minDF, max_df=maxDF)
        Vect.fit(X)
        X_train_WoR_dtm = Vect.transform(X)
        X_train_Features = Vect.get_feature_names()
        if include_rules == 'yes':
            df_notPreprocessed = pd.read_csv(path_to_nonPreprocessedDataset,  encoding = "ISO-8859-1")
            X_train_dtm, X_train_Features = self.apply_gen_rules_features_ngrams(df_notPreprocessed["tweet_text"], X_train_WoR_dtm, X_train_Features)
        else:
            X_train_dtm = X_train_WoR_dtm
        # removing duplicates from the sequences generated from text data
        X_train_dtm, indices_uniq = np.unique(X_train_dtm.toarray(), return_index=True, axis=0)
        return X_train_dtm, indices_uniq, Vect

    def attach_date(self, fname):
        created_at = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        fname = created_at + fname
        return fname


    def add_label(self, twt):
        output = []
        for i, s in zip(twt.index, twt):
            output.append(TaggedDocument(s, ["tweet_" + str(i)]))
        return output

    def load_word2vec_model(self, path, tokenizer, df, embed_dim=300):
        if "glove".lower() in path.lower().split("/")[2]:
            with open(path, 'r', encoding='UTF-8') as f:

                words = set()
                # list(tokenizer.word_index.keys())
                word_to_vec_map = {}
                for line in f:
                    w_line = line.split()
                    curr_word = w_line[0]
                    if curr_word in list(tokenizer.word_index.keys()):
                        word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
        elif "word2vec".lower() in path.lower().split("/")[2]:
            tokenized_tweet = df['tweet_text'].apply(lambda x: x.split())
            word_to_vec_map = gensim.models.Word2Vec(
                tokenized_tweet,
                vector_size=embed_dim,  # desired no. of features/independent variables
                window=5,  # context window size
                min_count=2,  # Ignores all words with total frequency lower than 2.
                sg=1,  # 1 for skip-gram model
                hs=0,
                negative=10,  # for negative sampling
                workers=32,  # no.of cores
                seed=34)
            word_to_vec_map.train(tokenized_tweet, total_examples=len(df['tweet_text']), epochs=20)
            word_to_vec_map = word_to_vec_map.wv
        elif "google".lower() in path.lower().split("/")[2]:

            word_to_vec_map = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(path, binary=True)
        elif "doc2vec".lower() in path.lower().split("/")[2]:

            tokenized_tweet = df['tweet_text'].apply(lambda x: x.split())
            labeled_tweets = self.add_label(tokenized_tweet)  # label all the tweets
            word_to_vec_map = gensim.models.Doc2Vec(dm=1,  # dm = 1 for ‘distributed memory’ model
                                                    dm_mean=1,  # dm_mean = 1 for using mean of the context word vectors
                                                    vector_size=embed_dim,  # no. of desired features
                                                    window=5,
                                                    # width of the context window
                                                    negative=7,  # if > 0 then negative sampling will be used
                                                    min_count=5,
                                                    # Ignores all words with total frequency lower than 5.
                                                    workers=32,  # no. of cores
                                                    alpha=0.1,  # learning rate
                                                    seed=23,  # for reproducibility
                                                    )
            word_to_vec_map.build_vocab([i for i in tqdm(labeled_tweets)])
            word_to_vec_map.train(labeled_tweets, total_examples=len(df['tweet_text']), epochs=15)
        elif "crisisNLP2vec".lower() in path.lower().split("/")[2]:
            word_to_vec_map = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(path, binary=True)

        return word_to_vec_map

    def maxLengthOfTweet(self, sequences):
        # Identify max length of reviews
        MAX_SEQUENCE_LENGTH = 0
        for tweet_number in range(len(sequences)):
            length = len(sequences[tweet_number])
            if (length) > (MAX_SEQUENCE_LENGTH):
                MAX_SEQUENCE_LENGTH = length
        return MAX_SEQUENCE_LENGTH


    def ruleString(self):
        rule = ""
        for i in range(1, 19):
            rule += "rule" + str(i)
            rule += " "
        return rule

    def dataToSequences(self, df, include_rules=False):
        tokenizer = Tokenizer()
        if include_rules:
            rule = self.ruleString()
            tokenizer.fit_on_texts(df["tweet_text"].append(pd.Series(rule)))
        else:
            tokenizer.fit_on_texts(df["tweet_text"])
        sequences = tokenizer.texts_to_sequences(df["tweet_text"])
        MAX_SEQUENCE_LENGTH = self.maxLengthOfTweet(sequences)
        train_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        return train_data, tokenizer, MAX_SEQUENCE_LENGTH

    def embeddnig_matrix(self, fun_word_to_vec_map, tokenizer, emb_matrix, include_rules=False):
        for word, index in tokenizer.word_index.items():
            try:
                embedding_vector = fun_word_to_vec_map[word]
                if embedding_vector is not None:
                    emb_matrix[index] = embedding_vector
            except KeyError:
                if include_rules:
                    rule = self.ruleString()
                    if word in rule:
                        emb_matrix[index] = np.ones(emb_matrix.shape[1]) / emb_matrix.shape[1]
                continue
        return emb_matrix

    def word_vector_avg(self, inv_tokenizer, fun_word_to_vec_map, tw_sequence, size):
        vec = np.zeros(size)
        count = 0
        columnNameList = []
        for i in range(18):
            columnName = "rule" + str(i + 1)
            columnNameList.append(columnName)
        for seq in tw_sequence:
            try:
                if inv_tokenizer[seq] in columnNameList:
                    embedding_vector = np.ndarray(shape=(size,))
                    embedding_vector[:] = 1/size
                else:
                    embedding_vector = fun_word_to_vec_map[inv_tokenizer[seq]]
                if embedding_vector is not None:
                    vec += embedding_vector
                else:
                    vec += np.zeros(size)
                count += 1.
            except KeyError:  # handling the case where the token is not in vocabulary
                continue
        if count != 0:
            vec /= count
        return vec

    def inv_tokenizer(self, tokenizer):
        inv_tokenizer = {v: k for k, v in tokenizer.word_index.items()}
        return inv_tokenizer

    def word2vec_Vect(self, path, df, embed_dim=300, path_to_nonPreprocessedDataset = "", include_rules=False):
        X_train_WoR_dtm, tokenizer, MAX_SEQUENCE_LENGTH = self.dataToSequences(df, include_rules)
        #     vocab_flat_list = list(tokenizer.word_index.keys())
        X_train_Features = ["_Not_Apply_"]
        if include_rules == 'yes':
            df_notPreprocessed = pd.read_csv(path_to_nonPreprocessedDataset,  encoding = "ISO-8859-1")
            X_train_dtm, _ = self.apply_gen_rules_features_word2vec(df_notPreprocessed["tweet_text"], X_train_WoR_dtm, tokenizer, X_train_Features)
            X_train_dtm = X_train_dtm.toarray()
            MAX_SEQUENCE_LENGTH += 18
        else:
            X_train_dtm = X_train_WoR_dtm
        # removing duplicates from the sequences generated from text data
        X_train_dtm, indices_uniq = np.unique(X_train_dtm, return_index=True, axis=0)

        inv_tokenizer = self.inv_tokenizer(tokenizer)
        vocab_len = len(tokenizer.word_index) + 1
        fun_word_to_vec_map = self.load_word2vec_model(path, tokenizer, df, embed_dim)
        vector = []
        for tw_sequence in X_train_dtm:
            vec = self.word_vector_avg(inv_tokenizer, fun_word_to_vec_map, tw_sequence, embed_dim)
            vector.append(vec)
        X_train_dtm = np.array(vector)

        return X_train_dtm, indices_uniq, X_train_Features, tokenizer, fun_word_to_vec_map, X_train_dtm.shape, vocab_len, MAX_SEQUENCE_LENGTH



    # X_train_dtm_word2vec, X_train_Features_word2vec, vect, dim, vocab_len, embed_vector_len, emb_matrix, MAX_SEQUENCE_LENGTH = \
    #     word2vec_Vect(path, df, embed_dim=300, include_rules=False)

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
    def generate_Features(self, X, y_binary, path_to_nonPreprocessedDataset = "", vect_type='simple', path2word2vecModel = "", embed_dim = 300, minDF=1, \
                          maxDF=1.0, include_rules = 'no', featureTypeFolderLabel=None):
        '''Generating Features from data
    -Takes attributes,responses, vect_type(simple by default), minDF, maxDF'''
        # generating name for the dataset file

        pBar = My_progressBar('Generating Features:', 6)
        if vect_type.lower() not in ["word2vec", "glove_word2vec", "word2vec_glove", "gloveword2vec", "word2vecglove", \
                                     "doc2vec", "glove" , "google", "google_news", "googlenews", "GoogleNews-vectors-negative300", "googleword2vec", "google_word2vec"]:
            self.pathToVectFolder = '../' + featureTypeFolderLabel + '_vect/'
            if self.pathToVectFolder == None:
                print("Please Provide the path to the folder for saving vectorizers")
                exit(0)
            self.create_Folder(self.pathToVectFolder)
            self.pathToVectFolder = self.pathToVectFolder
            save_vect = Save_Load_Model(self.pathToVectFolder)
            X = X["tweet_text"]
            if vect_type == 'simple':
                temp_label = 'CountVect'
                func_handler = self.simple_CountVect
            elif vect_type in ['tfidf', 'tf-idf', 'TfIdf', 'Tf-Idf', 'Tf-idf', 'Tfidf']:
                temp_label = 'TfIdfVect'
                func_handler = self.tfIdf_Vect
            if include_rules == 'yes':
                temp_label = temp_label + '_WRul'
            else:
                temp_label = temp_label + '_WoRul'
            for i in range(1, 4):
                for j in range(i, 4):
                    start_time = time.time()
                    if i == j == 1:
                        X_train_dtm_uni, indices_uniq, vect = \
                            func_handler(X, path_to_nonPreprocessedDataset, i, j, minDF, maxDF, include_rules)  # temp_label represents vectorizer type i.e., simple or tfidf

                        X_train_Features_uni = vect.get_feature_names()
                        y_binary_uni = np.take(y_binary, indices_uniq)
                        uniLabel = '_Unigrams_' + temp_label + '_Freq-' + str(X_train_dtm_uni.shape[1])
                        uniLabel = self.attach_date(uniLabel)
                        save_vect.save_Model(vect, uniLabel, uniLabel)

                    elif (i == 1) & (j == 2):
                        X_train_dtm_uniBi,  indices_uniq, vect = \
                            func_handler(X, path_to_nonPreprocessedDataset, i, j, minDF, maxDF, include_rules)
                        X_train_Features_uniBi = vect.get_feature_names()
                        y_binary_uniBi = np.take(y_binary, indices_uniq)
                        uniBiLabel = "_UniAndBigrams_" + temp_label + '_Freq-' + str( X_train_dtm_uniBi.shape[1])
                        uniBiLabel = self.attach_date(uniBiLabel)
                        save_vect.save_Model(vect, uniBiLabel, uniBiLabel)
                    elif (i == 1) & (j == 3):
                        X_train_dtm_uniBiTri,  indices_uniq, vect = \
                            func_handler(X, path_to_nonPreprocessedDataset, i, j, minDF, maxDF, include_rules)
                        X_train_Features_uniBiTri = vect.get_feature_names()
                        y_binary_uniBiTri = np.take(y_binary, indices_uniq)
                        uniBiTriLabel = "_UniBiAndTrigrams_" + temp_label + '_Freq-' + str(X_train_dtm_uniBiTri.shape[1])
                        uniBiTriLabel = self.attach_date(uniBiTriLabel)
                        save_vect.save_Model(vect, uniBiTriLabel, uniBiTriLabel)
                    elif (i == 2) & (j == 2):
                        X_train_dtm_bi,  indices_uniq, vect = \
                            func_handler(X, path_to_nonPreprocessedDataset, i, j, minDF, maxDF, include_rules)
                        X_train_Features_bi = vect.get_feature_names()
                        y_binary_bi = np.take(y_binary, indices_uniq)
                        biLabel = "_Bigrams_" + temp_label + '_Freq-' + str(X_train_dtm_bi.shape[1])
                        biLabel = self.attach_date(biLabel)
                        save_vect.save_Model(vect, biLabel, biLabel)
                    elif (i == 2) & (j == 3):
                        X_train_dtm_biTri,  indices_uniq, vect = \
                            func_handler(X, path_to_nonPreprocessedDataset, i, j, minDF, maxDF, include_rules)
                        X_train_Features_biTri = vect.get_feature_names()
                        y_binary_biTri = np.take(y_binary, indices_uniq)
                        biTriLabel = "_BiTrigrams_" + temp_label + '_Freq-' + str(X_train_dtm_biTri.shape[1])
                        biTriLabel = self.attach_date(biTriLabel)
                        save_vect.save_Model(vect, biTriLabel, biTriLabel)
                    elif (i == 3) & (j == 3):
                        X_train_dtm_tri,  indices_uniq, vect = \
                            func_handler(X, path_to_nonPreprocessedDataset, i, j, minDF, maxDF, include_rules)
                        X_train_Features_tri = vect.get_feature_names()
                        y_binary_tri = np.take(y_binary, indices_uniq)
                        triLabel = "_Trigrams_" + temp_label + '_Freq-' + str(X_train_dtm_tri.shape[1])
                        triLabel = self.attach_date(triLabel)
                        save_vect.save_Model(vect, triLabel, triLabel)
                    pBar.call_to_progress(start_time)

            encap = {'unigrams': {'dtm': X_train_dtm_uni, 'header': X_train_Features_uni, 'specs': uniLabel, 'y': y_binary_uni},
                     'uniBigrams': {'dtm': X_train_dtm_uniBi, 'header': X_train_Features_uniBi, 'specs': uniBiLabel, 'y':y_binary_uniBi},
                     'uniBiTrigrams': {'dtm': X_train_dtm_uniBiTri, 'header': X_train_Features_uniBiTri, 'specs': uniBiTriLabel, 'y':y_binary_uniBiTri},
                     'bigrams': {'dtm': X_train_dtm_bi, 'header': X_train_Features_bi, 'specs': biLabel, 'y':y_binary_bi},
                     'biTrigrams': {'dtm': X_train_dtm_biTri, 'header': X_train_Features_biTri, 'specs': biTriLabel, 'y':y_binary_biTri},
                     'trigrams': {'dtm': X_train_dtm_tri, 'header': X_train_Features_tri, 'specs': triLabel, 'y':y_binary_tri}}
        elif vect_type.lower() in ["word2vec", "glove_word2vec", "word2vec_glove", "gloveword2vec", "word2vecglove", \
                                   "doc2vec", "glove" , "google", "google_news", "googlenews", "GoogleNews-vectors-negative300", "googleword2vec", "google_word2vec"]:

            self.pathToVectFolder = '../' + featureTypeFolderLabel + '_vect/'
            self.pathToModelFolder = '../' + featureTypeFolderLabel + '_preTrainedModel/'
            if self.pathToVectFolder == None and self.pathToModelFolder == None:
                print("Please Provide the path to the folder for saving vectorizers or wrod2vec models.")
                exit(0)
            self.create_Folder(self.pathToVectFolder)
            self.create_Folder(self.pathToModelFolder)
            self.pathToVectFolder = self.pathToVectFolder
            self.pathToModelFolder = self.pathToModelFolder
            print(self.pathToModelFolder, self.pathToVectFolder)
            save_vect = Save_Load_Model(self.pathToVectFolder)
            save_word2vecModel = Save_Load_Model(self.pathToModelFolder)
            func_handler = self.word2vec_Vect

            # word2vec
            temp_label = 'word2vec'
            path2word2vecModel = 'dummyPath/word2vecModels/word2vec'
            if include_rules == 'yes':
                temp_label = temp_label + '_Vect_WRul'
            else:
                temp_label = temp_label + '_Vect_WoRul'
            X_train_dtm_word2vec, indices_uniq, X_train_Features_word2vec, vect_word2vec, fun_word_to_vec_map_word2vec, dim_word2vec, vocab_len_word2vec, MAX_SEQUENCE_LENGTH_word2vec = \
                func_handler(path2word2vecModel, X, embed_dim, path_to_nonPreprocessedDataset, include_rules)  # temp_label represents vectorizer type i.e., simple or tfidf
            y_binary_word2vec = np.take(y_binary, indices_uniq)
            word2vecLabel_word2vec =   "_" + temp_label + "_Max-Len-Seq-" + str(MAX_SEQUENCE_LENGTH_word2vec) + '_Freq-' + str(dim_word2vec[1])
            word2vecLabel_word2vec = self.attach_date(word2vecLabel_word2vec)
            save_vect.save_Model(vect_word2vec, word2vecLabel_word2vec, word2vecLabel_word2vec)
            save_word2vecModel.save_Model(fun_word_to_vec_map_word2vec, word2vecLabel_word2vec.replace('Vect', 'Model'), word2vecLabel_word2vec.replace('Vect', 'Model'))

            # doc2vec
            temp_label = 'doc2vec'
            path2word2vecModel = 'dummyPath/word2vecModels/doc2vec'
            if include_rules == 'yes':
                temp_label = temp_label + '_Vect_WRul'
            else:
                temp_label = temp_label + '_Vect_WoRul'
            X_train_dtm_doc2vec, indices_uniq, X_train_Features_doc2vec, vect_doc2vec, fun_word_to_vec_map_doc2vec, dim_doc2vec, vocab_len_doc2vec, MAX_SEQUENCE_LENGTH_doc2vec = \
                func_handler(path2word2vecModel, X, embed_dim, path_to_nonPreprocessedDataset, include_rules)  # temp_label represents vectorizer type i.e., simple or tfidf
            y_binary_doc2vec = np.take(y_binary, indices_uniq)
            word2vecLabel_doc2vec =   "_" + temp_label + "_Max-Len-Seq-" + str(MAX_SEQUENCE_LENGTH_doc2vec) + '_Freq-' + str(dim_doc2vec[1])
            # put sleep to prevent word2vecLabel_word2vec == word2vecLabel_doc2vec
            time.sleep(1)
            word2vecLabel_doc2vec = self.attach_date(word2vecLabel_doc2vec)

            save_vect.save_Model(vect_doc2vec, word2vecLabel_doc2vec, word2vecLabel_doc2vec)
            save_word2vecModel.save_Model(fun_word_to_vec_map_doc2vec, word2vecLabel_doc2vec.replace('Vect', 'Model'), word2vecLabel_doc2vec.replace('Vect', 'Model'))

            # glove2vec
            embed_dim_glove2vec = 200 # because glove2vec has dimention of 200 others have 300
            temp_label = 'glove'
            path2word2vecModel = '../word2vecModels/glove.twitter.27B.200d.txt'
            if include_rules == 'yes':
                temp_label = temp_label + '_Vect_WRul'
            else:
                temp_label = temp_label + '_Vect_WoRul'
            X_train_dtm_glove2vec, indices_uniq, X_train_Features_glove2vec, vect_glove2vec, fun_word_to_vec_map_glove2vec, dim_glove2vec, vocab_len_glove2vec, MAX_SEQUENCE_LENGTH_glove2vec = \
                func_handler(path2word2vecModel, X, embed_dim_glove2vec, path_to_nonPreprocessedDataset, include_rules)  # temp_label represents vectorizer type i.e., simple or tfidf
            y_binary_glove2vec = np.take(y_binary, indices_uniq)
            word2vecLabel_glove2vec =   "_" + temp_label + "_Max-Len-Seq-" + str(MAX_SEQUENCE_LENGTH_glove2vec) + '_Freq-' + str(dim_glove2vec[1])
            word2vecLabel_glove2vec = self.attach_date(word2vecLabel_glove2vec)
            save_vect.save_Model(vect_glove2vec, word2vecLabel_glove2vec, word2vecLabel_glove2vec)
            save_word2vecModel.save_Model(fun_word_to_vec_map_glove2vec, word2vecLabel_glove2vec.replace('Vect', 'Model'), word2vecLabel_glove2vec.replace('Vect', 'Model'))

            # google2vec
            temp_label = 'google2vec'
            path2word2vecModel = '../word2vecModels/GoogleNews-vectors-negative300.bin'
            if include_rules == 'yes':
                temp_label = temp_label + '_Vect_WRul'
            else:
                temp_label = temp_label + '_Vect_WoRul'
            X_train_dtm_google2vec, indices_uniq, X_train_Features_google2vec, vect_google2vec, fun_word_to_vec_map_google2vec, dim_google2vec, vocab_len_google2vec, MAX_SEQUENCE_LENGTH_google2vec = \
                func_handler(path2word2vecModel, X, embed_dim, path_to_nonPreprocessedDataset, include_rules)  # temp_label represents vectorizer type i.e., simple or tfidf
            y_binary_google2vec = np.take(y_binary, indices_uniq)
            word2vecLabel_google2vec =   "_" + temp_label + "_Max-Len-Seq-" + str(MAX_SEQUENCE_LENGTH_google2vec) + '_Freq-' + str(dim_google2vec[1])
            word2vecLabel_google2vec = self.attach_date(word2vecLabel_google2vec)
            save_vect.save_Model(vect_google2vec, word2vecLabel_google2vec, word2vecLabel_google2vec)
            save_word2vecModel.save_Model(fun_word_to_vec_map_google2vec, word2vecLabel_google2vec.replace('Vect', 'Model'), word2vecLabel_google2vec.replace('Vect', 'Model'))



            # crisisNLP2vec
            temp_label = 'crisisNLP2vec'
            path2word2vecModel = '../word2vecModels/crisisNLP2vec.bin'
            if include_rules == 'yes':
                temp_label = temp_label + '_Vect_WRul'
            else:
                temp_label = temp_label + '_Vect_WoRul'
            X_train_dtm_crisisNLP2vec, indices_uniq, X_train_Features_crisisNLP2vec, vect_crisisNLP2vec, fun_word_to_vec_map_crisisNLP2vec, dim_crisisNLP2vec, vocab_len_crisisNLP2vec, MAX_SEQUENCE_LENGTH_crisisNLP2vec = \
                func_handler(path2word2vecModel, X, embed_dim, path_to_nonPreprocessedDataset, include_rules)  # temp_label represents vectorizer type i.e., simple or tfidf
            y_binary_crisisNLP2vec = np.take(y_binary, indices_uniq)
            word2vecLabel_crisisNLP2vec =   "_" + temp_label + "_Max-Len-Seq-" + str(MAX_SEQUENCE_LENGTH_crisisNLP2vec) + '_Freq-' + str(dim_crisisNLP2vec[1])
            word2vecLabel_crisisNLP2vec = self.attach_date(word2vecLabel_crisisNLP2vec)
            save_vect.save_Model(vect_crisisNLP2vec, word2vecLabel_crisisNLP2vec, word2vecLabel_crisisNLP2vec)
            save_word2vecModel.save_Model(fun_word_to_vec_map_crisisNLP2vec, word2vecLabel_crisisNLP2vec.replace('Vect', 'Model'), word2vecLabel_crisisNLP2vec.replace('Vect', 'Model'))



            encap = {'word2vec': {'dtm': X_train_dtm_word2vec, 'header': X_train_Features_word2vec,'specs': word2vecLabel_word2vec, 'y':y_binary_word2vec},
                     'doc2vec': {'dtm': X_train_dtm_doc2vec, 'header': X_train_Features_doc2vec,'specs': word2vecLabel_doc2vec, 'y':y_binary_doc2vec},
                     'glove2vec': {'dtm': X_train_dtm_glove2vec, 'header': X_train_Features_glove2vec,'specs': word2vecLabel_glove2vec, 'y':y_binary_glove2vec},
                     'google2vec': {'dtm': X_train_dtm_google2vec, 'header': X_train_Features_google2vec,'specs': word2vecLabel_google2vec, 'y':y_binary_google2vec},
                     'crisisNLP2vec': {'dtm': X_train_dtm_crisisNLP2vec, 'header': X_train_Features_crisisNLP2vec,'specs': word2vecLabel_crisisNLP2vec, 'y':y_binary_crisisNLP2vec}
                     }
        return encap


script_start = time.time()  # script start point
print("\nscritp completed")
Total_time = time.time() - script_start
print("\nTotal time for script completion :" + str(datetime.timedelta(seconds=int(Total_time))))
