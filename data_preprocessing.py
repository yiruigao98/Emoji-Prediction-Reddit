import pandas as pd
import string
import numpy as np
import csv 
import re
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import gensim
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

tkwp = WordPunctTokenizer()
stop_words = stopwords.words('english')


# Read texts from the file:
def read_files(text_path):
    df =pd.read_csv(text_path, header=0, engine="python")
    return df


# Build a corpus by preprocessing:
def normalize_text(text):
    # remove special characters:
    text = list(map(lambda token: re.sub('[^0-9A-Za-z!?]+', '', token), text))
    text = list(map(lambda token: re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', token), text))
    text = list(map(lambda token: re.sub('www|http', '', token), text))
    text = list(filter(lambda token: token, text))
    return text


# Word2vec preprocessing:
def word2vec_pre(df):
    df['Content'] = df.Content.str.lower().str.strip()
    # Separate sentences:
    df['Sentences'] = df.Content.str.split('.')
    # Tokenization:
    df['Tokenized_Sentence'] = list(map(lambda sen: list(map(nltk.word_tokenize, sen)), df.Sentences))
    # Normalize texts:
    df['Tokenized_Sentence'] = list(map(lambda sen: list(map(normalize_text, sen)), df.Tokenized_Sentence))
    # Remove empty lists:
    df['Tokenized_Sentence'] = list(map(lambda sen: list(filter(lambda lst: lst, sen)), df.Tokenized_Sentence))


# Remove stop words:
def remove_stop(df):
    df['No_stop_Content'] = list(map(lambda text: [word for word in text if word not in stop_words], df['Tokenized_Content']))


# Lemmatize:
def lemma(df):
    lemm = nltk.stem.WordNetLemmatizer()
    df['Lemmatized_Content'] = list(map(lambda sen: list(map(lemm.lemmatize, sen)), df.No_stop_Content))


# Stemming:
def stem(df):
    p_stemmer = nltk.stem.PorterStemmer()
    df['Stemmed_Content'] = list(map(lambda sen: list(map(p_stemmer.stem, sen)), df.Lemmatized_Content))


# Vectorization:
def vectorize(df):
    # Create a dictionary to store the words that are going to be used as features:
    dic = Dictionary(documents=df.Stemmed_Content.values)
    print("Found %d words."%len(dic.values()))
    dic.filter_extremes(no_above=0.5, no_below=3)
    # Reindexes:
    dic.compactify()
    print("Left with %d words."%len(dic.values()))
    # Bag of Word:
    df['Bow'] = list(map(lambda text: dic.doc2bow(text), df.Stemmed_Content))
    return dic


# LDA preprocessing:
def LDA_pre(df):
    df['Content'] = df.Content.str.lower().str.strip()
    df['Tokenized_Content'] = list(map(nltk.word_tokenize, df.Content))
    df['Tokenized_Content'] = list(map(normalize_text, df.Tokenized_Content))
    remove_stop(df)
    lemma(df)
    stem(df)
    dic = vectorize(df)
    return dic


# Scale and normalize the two personal data columns:
def norm(df):
    x = df[['Comment_karma', 'Link_karma']]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled)
    df['Ckarma_scaled'] = list(x_scaled[0])
    df['Lkarma_scaled'] = list(x_scaled[1])
    return df


# One hot encoding to personality:
def onehotper(df):
    x = df[['Personality']]
    le = LabelEncoder()
    ohe = OneHotEncoder(sparse=False)
    X_2 = x.apply(le.fit_transform)
    x_ohe = ohe.fit_transform(X_2)

    df2 = pd.DataFrame(x_ohe, columns=['P0','P1','P2','P3','P4','P5','P6','P7','P8','P9'])
    df = pd.concat([df, df2], axis=1)
    return df


# Generate n-grams:
def generate_ngrams(tokens, n):
    ngrams = zip(*[tokens[i:] for i in range(n)])
    ngram_list = [" ".join(ngram) for ngram in ngrams]
    return ngram_list


# Calculate tf-idf:
def computeTF(ngram_list):
    TFdic = {}
    for gram in ngram_list:
        if gram in TFdic:
            TFdic[gram] += 1
        else:
            TFdic[gram] = 1
    for gram in TFdic:
        TFdic[gram] = TFdic[gram]/len(ngram_list)
    return TFdic

def computeCount(ngram_lists, TFdic_all):
    count_dic = {}
    ngram_list_id = 0
    for TFdic in TFdic_all:
        for gram in ngram_lists[ngram_list_id]:
            if gram in count_dic:
                count_dic[gram] += 1
            else:
                count_dic[gram] = 1
    return count_dic

def computeIDF(count_dic, N):
    IDFdic = {}
    for gram in count_dic:
        IDFdic[gram] = math.log(N/count_dic[gram])
    return IDFdic


#------------------------------------------------------------#
# Use a traditional way without embedding to formalize features:
# Select features:
def select_feature(IDFdic):
    sorted_d = sorted(IDFdic.items(), key = lambda x: x[1])
    feature_list = sorted_d.keys()[:1000]
    return feature_list

# Form matrix of features:
def form_row_feature(TFdic, ngram_list, feature_list):
    X_row = []
    for feature in feature_list:
        if TFdic.get(feature, None) != None:
            X_row.append(TFdic.get(feature)*len(ngram_list))
        else:
            X_row.append(0)          
    return X_row





