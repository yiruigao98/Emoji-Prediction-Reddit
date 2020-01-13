import pandas as pd
import numpy as np
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import Word2Vec

# LDA:
def LDA_train(df, dic):
    corpus = df.Bow
    num_topics = 50
    LDAmodel = LdaMulticore(corpus=corpus,
                id2word=dic, 
                num_topics=num_topics)
    return LDAmodel


def LDA_features(model, texts):
    topic_importance = model.get_document_topics(texts, minimum_probability = 0)
    topic_importance = np.array(topic_importance)
    return topic_importance[:, 1]


# Word2vec:
def Word2vec_train(df):
    sentences = []
    for sen_group in df.Tokenized_Sentence:
        sentences.extend(sen_group)
    print("Number of sentences is %d"%len(sentences))
    print("Number of texts is %d"%len(df))
    # Parameters:
    num_features = 100
    min_count  = 3
    window = 6
    downsampling = 1e-3
    W2Vmodel = Word2Vec(sentences=sentences, 
                sg=1,
                hs=0,
                size=num_features,
                window=window,
                min_count=min_count,
                sample=downsampling,
                negative=5,
                iter=6)
    return W2Vmodel


def W2V_features(model, texts):
    words = np.concatenate(texts)
    index2word_set = set(model.wv.vocab.keys())
    featureVec = np.zeros(model.vector_size, dtype = "float32")
    num_words = 0
    for word in words:
        if word in index2word_set:
            featureVec = np.add(featureVec, model[word])
            num_words += 1
    if num_words > 0:
        featureVec = np.divide(featureVec, num_words)
    return featureVec