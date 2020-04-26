import numpy as np
import pandas as pd
import nltk
import os

from textblob import TextBlob

import collections
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline = False, world_readable = True)
from sklearn.preprocessing import MinMaxScaler
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import *

##

def preprocessing(ent):

    ent = str(ent)

    ent = ent.lower()
    ent = ent.replace("\n", "")
    ent = re.sub(r'^0-9a-z #+_', '', ent) #Removing digits
    ent = re.sub(r'^https?:\/\/.*[\r\n]*', '', ent) #Removing URLs

    punctuations = '''!()-[]{};:'"\,<>/?@#$%^&*_~'''

    for char in ent:
        if char in punctuations:
            ent = ent.replace(char, " ")

    ent = ent.strip()

    return ent

##

def remstopwords(string):

    string = str(string)

    wordsCorpus = set(nltk.corpus.words.words())
    wordsCorpus.add('.')

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(string)

    words = [word for word in tokens if word not in stop_words]
    words = [word for word in words if word in wordsCorpus or not word.isalpha]

    string = ' '.join(word for word in words)
    string = string.strip()

    return string

##

def lemmatize(string):

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(string)
    for ind in range(len(tokens)):

        tokens[ind] = lemmatizer.lemmatize(tokens[ind])

    string = ' '.join(word for word in tokens)

    return string

##

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

##

def normalize(feature):
    arr = data_dum[feature].values.reshape(-1,1)

    min_max_scaler = MinMaxScaler()
    arr_scaled = min_max_scaler.fit_transform(arr)

    data_dum[feature] = arr_scaled

##
