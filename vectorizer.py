from sklearn.feature_extraction.text import TfidfVectorizer

import gensim.downloader as api

from bs4 import BeautifulSoup

import numpy as np


tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2', sublinear_tf=True, ngram_range=(1, 3), max_features=1000,
                                   stop_words='english')


def preprocess(text):
    text = BeautifulSoup(' '.join(text.split()), 'html.parser').get_text()
    return text


def word2vec_avg(text):
    word2vec_model = api.load('word2vec-google-news-300')
    words = text.split()
    words = [word for word in words if word in word2vec_model.key_to_index]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return []


def get_tfidf_vectors(train_data, test_data):
    tfidf_vectorizer.fit_transform(train_data)

    train_feature_set = tfidf_vectorizer.transform(train_data).toarray()
    if test_data.empty:
        test_feature_set = None
    else:
        test_feature_set = tfidf_vectorizer.transform(test_data).toarray()
    return train_feature_set, test_feature_set


def get_word2vec_avg_vectors(train_data, test_data):
    train_feature_set = train_data.apply(lambda x: word2vec_avg(x))
    if test_data == None:
        test_feature_set = None
    else:
        test_feature_set = test_data.apply(lambda x: word2vec_avg(x))
    return train_feature_set, test_feature_set


def get_feature_vectors(train_data, test_data, method):
    train_data = train_data.apply(lambda x: preprocess(x))
    if test_data != None:
        test_data = test_data.apply(lambda x: preprocess(x))
    print(train_data)
    if method == "tfidf":
        return get_tfidf_vectors(train_data, test_data)
    if method == "word2vec":
        return get_word2vec_avg_vectors(train_data, test_data)