import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import pickle
from user_info import *


type_indicators = [ "IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) – Sensing (S)", 
                   "FT: Feeling (F) - Thinking (T)", "JP: Judging (J) – Perceiving (P)"  ]
b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]
# Personality type list:
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
unique_type_list = [x.lower() for x in unique_type_list]

# Preprocessing variables:
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
StopWords = stopwords.words("english")

tfizer = TfidfTransformer()
cntizer = CountVectorizer(analyzer = "word", 
                             max_features = 3000, 
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,  
                             max_df = 0.7,
                             min_df = 0.1) 


def display_data(data):
    print("Print the first 10 rows of the data:")
    print(data.head(10))
    print("Print the posts for the first two row of the data:")
    for p in data.head(2).posts.values:
        print(p.split("|||"))


def plot(data):
    cnt_types = data['type'].value_counts()
    plt.figure(figsize=(12,4))
    sns.barplot(cnt_types.index, cnt_types.values, alpha=0.8)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Types', fontsize=12)
    plt.show()


def get_type(row):
    t = row['type']
    I = N = T = J = 0
    if t[0] == 'I':
        I = 1
    elif t[0] == 'E':
        I = 0
    if t[1] == 'N':
        N = 1
    elif t[1] == 'S':
        N = 0    
    if t[2] == 'T':
        T = 1
    elif t[2] == 'F':
        T = 0
    if t[3] == 'J':
        J = 1
    elif t[3] == 'P':
        J = 0   
    return pd.Series( {'IE':I, 'NS':N , 'TF': T, 'JP': J }) 


def calc_fre(data):
    print ("Introversion (I) /  Extroversion (E):\t", data['IE'].value_counts()[0], " / ", data['IE'].value_counts()[1])
    print ("Intuition (N) – Sensing (S):\t\t", data['NS'].value_counts()[0], " / ", data['NS'].value_counts()[1])
    print ("Thinking (T) – Feeling (F):\t\t", data['TF'].value_counts()[0], " / ", data['TF'].value_counts()[1])
    print ("Judging (J) – Perceiving (P):\t\t", data['JP'].value_counts()[0], " / ", data['JP'].value_counts()[1])



def plot_dis(data):
    N = 4
    bot = (data['IE'].value_counts()[0], data['NS'].value_counts()[0], data['TF'].value_counts()[0], data['JP'].value_counts()[0])
    top = (data['IE'].value_counts()[1], data['NS'].value_counts()[1], data['TF'].value_counts()[1], data['JP'].value_counts()[1])

    ind = np.arange(N)    # the x locations for the groups
    width = 0.7      # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, bot, width)
    p2 = plt.bar(ind, top, width, bottom = bot)

    plt.ylabel('Count')
    plt.title('Distribution accoss types indicators')
    plt.xticks(ind, ('I/E',  'N/S', 'T/F', 'J/P',))
    plt.show()


def corr(data):
    print("The correlation of the data is: ")
    print(data[['IE','NS','TF','JP']].corr())
    cmap = plt.cm.RdBu
    corr = data[['IE','NS','TF','JP']].corr()
    plt.figure(figsize=(12,10))
    plt.title('Pearson Features Correlation', size=15)
    sns.heatmap(corr, cmap=cmap,  annot=True, linewidths=1)
    plt.show()


def translate_personality(p):
    return [b_Pers[l] for l in p]

def translate_back(p):
    s = ""
    for i,l in enumerate(p):
        s += b_Pers_list[i][l]
    return s

def test_Binarize(data):
    d = data.head(10)
    print(d.type)
    bin_personality = np.array([translate_personality(p) for p in d.type])
    print("Binarize MBTI list: \n%s"%bin_personality)


def preprocessing(data, no_stop_wprds = True, no_mbti_profile = True):
    list_personality = []
    list_posts = []
    len_data = len(data)
    i = 0
    for row in data.iterrows():
        i += 1
        if (i % 1000 == 0 or i == 1 or i == len_data):
            print("%s of %s rows" % (i, len_data))

        ##### Remove and clean comments
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        if no_stop_wprds:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in StopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
            
        if no_mbti_profile:
            for t in unique_type_list:
                temp = temp.replace(t, "")

        type_labelized = translate_personality(row[1].type)
        list_personality.append(type_labelized)
        list_posts.append(temp)

    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality


def vectorize(list_posts):
    X_cnt = cntizer.fit_transform(list_posts)
    X_tfitf = tfizer.fit_transform(X_cnt).toarray()
    return X_tfitf
    

def train_individual(X, list_posts, list_personality):
    for l in range(len(type_indicators)):
        print("%s..."%(type_indicators[l]))

        Y = list_personality[:,l]

        param_grid = {
            'n_estimators' : [ 200, 300],
            'learning_rate': [ 0.2, 0.3]
        }

        # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        # grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
        # grid_result = grid_search.fit(X, Y)

        seed = 7
        test_size = .25
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = seed)

        eval_set = [(X_test, y_test)]

        # XGBoost model:
        param = {}
        param['n_estimators'] = 200
        param['max_depth'] = 2
        param['nthread'] = 8
        param['learning_rate'] = 0.2
        model = XGBClassifier(**param)

        model.fit(X_train, y_train, early_stopping_rounds= 10, eval_metric="logloss", eval_set=eval_set, verbose=True)
        # Predictions:
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("* %s Accuracy: %.2f%%" % (type_indicators[l], accuracy*100))
        # Save the model:
        filename = "personal_model_%s.sav"%str(l)
        pickle.dump(model, open(filename, 'wb'))



def get_personality(username, reddit):
    
    # Concat all the posts of this user:
    submissions = get_submissions(username, reddit)
    comments = get_comments(username, reddit)
    # print(submissions)
    # print("************************************")
    # print(comments)
    post_list = []
    for s in submissions:
        post_list.append(s)
    for c in comments:
        post_list.append(c)
    posts = "".join(post_list)
    posts = posts.replace('\n','')
    print(posts)
    print(type(posts))
    my_posts = posts

    mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})

    my_posts, dummy = preprocessing(mydata)

    my_X_cnt = cntizer.transform(my_posts)
    my_X_tfidf =  tfizer.transform(my_X_cnt).toarray()

    result = []
    # Let's train type indicator individually
    for l in range(len(type_indicators)):
        print("%s ..." % (type_indicators[l]))
        
        model = pickle.load(open('personal_model_%s.sav'%str(l), 'rb'))
        # make predictions for my  data
        y_pred = model.predict(my_X_tfidf)
        result.append(y_pred[0])
        # print("* %s prediction: %s" % (type_indicators[l], y_pred))
    print("The result is: ", translate_back(result))
    personality = translate_back(result)
    
    return personality




if __name__ == "__main__":
    data = pd.read_csv("BYOtrain.csv")
    # display_data(data)
    # plot(data)
    data_type = data.join(data.apply(lambda row: get_type (row),axis=1))
    # display_data(data_type)
    calc_fre(data_type)
    # plot_dis(data_type)
    # corr(data_type)

    # Preprocess data:
    ## Test binarize:
    test_Binarize(data_type)
    list_posts, list_personality = preprocessing(data_type)
    # Vectorize:
    X_tfitf = vectorize(list_posts)
    features = list(enumerate(cntizer.get_feature_names()))
    print(features)

    for l in range(len(type_indicators)):
        print(type_indicators[l])
    print("MBTI 1st row: %s" % translate_back(list_personality[0,:]))
    print("Y: Binarized MBTI 1st row: %s" % list_personality[0,:])

    train_individual(X_tfitf, list_posts, list_personality)

    # reddit = praw.Reddit(client_id = "jojUq2jbLlQQnw", client_secret='1PooR7o5_XyJH60X7VJmFrAMX7s',
    #     user_agent='/u/yiruigao98')
    # submissions = get_submissions('chuggo_tuggans', reddit)
    # comments = get_comments('chuggo_tuggans', reddit)
    # post_list = []
    # for s in submissions:
    #     post_list.append(s)
    # for c in comments:
    #     post_list.append(c)
    # posts = "".join(post_list)
    # # posts = posts.replace('\n','')
    # print(type(posts))
    # my_posts = posts

    # # The type is just a dummy so that the data prep fucntion can be reused
    # mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})

    # my_posts, dummy  = preprocessing(mydata)

    # my_X_cnt = cntizer.transform(my_posts)
    # my_X_tfidf =  tfizer.transform(my_X_cnt).toarray()

    # result = []
    # # Let's train type indicator individually
    # for l in range(len(type_indicators)):
    #     print("%s ..." % (type_indicators[l]))
        
    #     model = pickle.load(open('personal_model_%s.sav'%str(l), 'rb'))
    #     # make predictions for my  data
    #     y_pred = model.predict(my_X_tfidf)
    #     result.append(y_pred[0])
    #     # print("* %s prediction: %s" % (type_indicators[l], y_pred))
    # print("The result is: ", translate_back(result))

    reddit = praw.Reddit(client_id = "jojUq2jbLlQQnw", client_secret='1PooR7o5_XyJH60X7VJmFrAMX7s',
        user_agent='/u/yiruigao98')
    print(get_personality('chuggo_tuggans', reddit))










