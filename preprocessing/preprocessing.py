
# coding: utf-8

# In[3]:


import pandas as pd
import time
import re
import datetime
from nltk.stem import PorterStemmer
from pywsd.utils import lemmatize_sentence
from nltk.corpus import wordnet as wn
# import preprocessor as p
# from autocorrect import spell
import csv
import sys
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from collections import Counter
import json
from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
script_start = time.time()
global steps
global dataset_Size
global count

# # Showing progress <--------------
def call_to_progress(start_time):
  global count
  global elapsed_time
  count = count + 1
  elapsed_time = time.time() - start_time
  if count % 500 == 0:
    progress(elapsed_time)
  if count == dataset_Size:
    progress(elapsed_time)
  
def init_prgoress_para():
  global count
  global steps
  global dataset_Size
  count = 0
  steps = dataset_Size//500
def progress(cal_time):
  bar_len = 50
  time = cal_time * (dataset_Size-count)
  status = str(datetime.timedelta(seconds=int(time)))
  filled_len = int(round(bar_len * count / float(dataset_Size)))
  percents = round(100.0 * count / float(dataset_Size), 1)
  bar = '=' * filled_len + '-' * (bar_len - filled_len)
  sys.stdout.write('[%s] [%s%s] [eta:%s]\r' % (bar, percents, '%', status))
  sys.stdout.flush()

# Showing progress -------------->

# ****************************************************************
# Removing Punctuations <--------------
def remove_punc(text):
  start_time = time.time()
  text = text.lower()
  tweet_tokenizer = TweetTokenizer()
  punctuation = list(string.punctuation)
  tokens = tweet_tokenizer.tokenize(text)
  clean_tokens = []
  for tok in tokens:
    if tok not in punctuation:
      clean_tokens.append(tok)
  text = token_str(clean_tokens)
  call_to_progress(start_time)
  return text
# Removing Punctuations -------------->

#****************************************************************
# stemming <------------
def stemming(text):
  start_time = time.time()
  tweet_tokenizer = TweetTokenizer()
  stemmer = PorterStemmer() 
  result = [stemmer.stem(i.lower()) for i in tweet_tokenizer.tokenize(text)]
  call_to_progress(start_time)
  return token_str(result)
# stemming ------------>
#****************************************************************


# converting tokens to string <------------
def token_str(tokens = []):
  return ' '.join(tokens)
# converting tokens to string ------------>
#****************************************************************

# converting list string <------------
def list_str(list1):
  return ' '.join(list1)
# converting list string  ------------>

#****************************************************************

# removing Non-ascii characters <-----------
def removeNonAscii(s):
  start_time = time.time()
  chunks = []
  for i in s: 
    if ord(i)<128:
      chunks.append(i)
  text=''.join(chunks)
  call_to_progress(start_time)
  return text
# removing Non-ascii characters ----------->

#****************************************************************

# removing stopwords <------------
def removeStopWords(text):
  #porter = PorterStemmer()
  start_time = time.time()
  stopword_list = set(stopwords.words('english'))
  tweet_tokenizer = TweetTokenizer()
  # result = [spell(porter.stem(i.lower())) for i in tweet_tokenizer.tokenize(doc) if i.lower() not in stopword_list]
  result = [i.lower() for i in tweet_tokenizer.tokenize(text) if i.lower() not in stopword_list]
  call_to_progress(start_time)
  return token_str(result)
# removing stopwords ------------->

#****************************************************************




#****************************************************************

# Num, URL, Menition, RT replacement <------------
def replace_Num_Url_Mention_RT(text):
  start_time = time.time()
  result = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)','_NUM_', text)
  result = re.sub(r'(?:(RT|rt) @ ?[\w_]+:?)','_RT_', result)
  result = re.sub(r'(?:@ ?[\w_]+)','_MENTIOM_', result)
  result = re.sub(r'http[s]? ?: ?//(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
                  '_URL_', result)
  call_to_progress(start_time)
  return result
# Num, URL, Menition, RT replacement ------------->
#****************************************************************

tweet_tokenizer = TweetTokenizer()
# fname = "disaster data.csv"
fname = "binary_dataset_Copy.csv"
path = "../datasets/"
dataset = pd.read_csv(path + fname, index_col= False, encoding = "ISO-8859-1")
print("Dataset Loaded Successfully")
dataset_Size = len(dataset)
print('size :')
print(dataset_Size)
# dataset = dataset.drop_duplicates(subset='tweet_text', keep='first')
# print('Duplicate tweets are removed:')
print('size :')
dataset_Size = len(dataset)
# print(dataset_Size)


print("\nRemoving non-Ascii Characters...")
init_prgoress_para()
dataset['tweet_text']= dataset['tweet_text'].apply(removeNonAscii)
# # dataset.to_csv("/home/sahibzada/Desktop/ipythonNB/ThImp/Datasets/specific labelled/remov_nonascii.csv")
print("\nNon-Ascii Characters Removed!")

dataset['tweet_text']= dataset['tweet_text'].apply(remove_punc)
# dataset.to_csv("Datasets/testing_remov_punc.csv")
print("Punctuations Removed")


print("\nRemoving Stopwords...")
init_prgoress_para()
dataset['tweet_text']= dataset['tweet_text'].apply(removeStopWords)
print("\nStop Words Removed!")


print("\nLemmatization started...")
init_prgoress_para()
dataset['tweet_text']= dataset['tweet_text'].apply(lemmatize_sentence)
dataset['tweet_text']= dataset['tweet_text'].apply(list_str)
print("\nLemmatization completed!")
#
print("\nURLs, Mentions, Retweets, Numbers replacement started...")
init_prgoress_para()
dataset['tweet_text'] = dataset['tweet_text'].apply(replace_Num_Url_Mention_RT)
# dataset = dataset.drop_duplicates(subset='tweet_text', keep='first')
# print('Duplicate tweets are removed:')
print("\nURLs, Mentions, Retweets, Numbers replacement completed!")
dataset.to_csv(path + 'preprocessed_' + fname)


# init_prgoress_para()
# print(dataset.head())
# dataset['tweet_text']= dataset['tweet_text'].apply(spell_check)
# dataset.to_csv("Datasets/TestingDatasets/Thursday_spellcheck.csv")
# print("Spell Corrected")
Total_time = time.time() - script_start
print("Total time for script completion" + str(datetime.timedelta(seconds=int(Total_time))))
