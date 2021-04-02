#!/usr/bin/env python.

"""
CS4248 Project
"""

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from textblob import TextBlob
import datetime as dt
import numpy as np
import pandas as pd
import scipy as sp
import string


def engineer_features(x_train):
    """
    Engineer features of training data
    :param x_train: DataFrame of training data with column names following column names of input csv files
    :return: sparse matrix of engineered features
    """
    x_train_df = pd.DataFrame(data=x_train)
    x_train_df['Posts'] = x_train['Input.posts'].apply(lambda x: BeautifulSoup(x, features="html.parser").get_text())
    features_df = pd.DataFrame(index=x_train_df.index)
    
    # == Thread only features ==
    # 1. If thread was started by an anonymous user
    # TODO but not sure
    # 2. If thread was marked as approved or deleted
    features_df['f2a'] = x_train_df['ApprovalTime'].apply(lambda x: 1 if x != '' else 0)
    features_df['f2b'] = x_train_df['RejectionTime'].apply(lambda x: 1 if x != '' else 0)
    # 3. Forum ID
    # TODO but not sure
    # 4. Start time of thread
    features_df['f4'] = x_train_df['CreationTime'].apply(lambda x: dt.datetime.strptime(x[4:-8] + x[-4:], '%b %d %H:%M:%S %Y').timestamp())
    # 5. Time of last post in thread
    # TODO
    # 6. Total number of posts in thread
    # TODO
    # 7. If thread title contains the words "lecture[s]"
    list7 = ['lecture', 'lectures']
    features_df['f7'] = x_train_df['Input.threadtitle'].apply(lambda x: 1 if any([w in x for w in list7]) else 0)
    # 8. If thread title contains the words "assignment[s]", "quiz[zes]", "grade[s]", "project[s]", "exam[s]"
    list8 = ['assignment', 'assignments', 'quiz', 'quizzes', 'grade', 'grades,' 'project', 'projects', 'exam', 'exams', 'reading', 'readings']
    features_df['f8'] = x_train_df['Input.threadtitle'].apply(lambda x: 1 if any([w in x for w in list8]) else 0)
    
    # == Aggregated post features (presense of keywords) ==
    # 9. Total number of votes for each individual post
    # TODO
    # 10. Mean and variance of posting times of individual posts in thread
    # TODO
    # 11. Mean of time difference between between the posting times of individual posts and the closest course landmark. A course landmark is the deadline of an assignment, exam or project.
    # TODO
    # 12. Count of occurrences of assessment related words "grade[s]", "exam[s]", "assignment[s]", "quiz[zes]", "reading[s]", "project[s]"
    list12 = ['assignment', 'assignments', 'quiz', 'quizzes', 'grade', 'grades,' 'project', 'projects', 'exam', 'exams', 'reading', 'readings']
    features_df['f12'] = x_train_df['Posts'].apply(lambda x: sum([x.count(w) for w in list12]))
    # 13. Count of occurences of words indicating technical problems "problem[s]", "error[s]"
    list13 = ['problem', 'problems', 'error', 'errors']
    features_df['f13'] = x_train_df['Posts'].apply(lambda x: sum([x.count(w) for w in list13]))
    # 14. Count of occurences of thread conclusive words "thank you", "thank[s]", "appreciate"
    list14 = ['thank you', 'thank', 'thanks', 'appreciate']
    features_df['f14'] = x_train_df['Posts'].apply(lambda x: sum([x.count(w) for w in list14]))
    # 15. Count of occurences of words like "request", "submit", "suggest"
    list15 = ['request', 'submit', 'suggest']
    features_df['f14'] = x_train_df['Posts'].apply(lambda x: sum([x.count(w) for w in list14]))
    
    # == POS Tags of thread title and posts (labelled a and b respectively) ==
    # 16. word count
    features_df['16a'] = x_train_df['Input.threadtitle'].apply(lambda x: len(x.split()))
    features_df['16b'] = x_train_df['Posts'].apply(lambda x: len(x.split()))
    # 17. character count
    features_df['17a'] = x_train_df['Input.threadtitle'].apply(len)
    features_df['17b'] = x_train_df['Posts'].apply(len)
    # 18. word density
    features_df['18a'] = features_df['17a'] / (features_df['16a'] + 1)
    features_df['18b'] = features_df['17b'] / (features_df['16b'] + 1)
    # 19. punctuation count
    features_df['19a'] = x_train_df['Input.threadtitle'].apply(lambda x: len("".join(p for p in x if p in string.punctuation)))
    features_df['19b'] = x_train_df['Posts'].apply(lambda x: len("".join(p for p in x if p in string.punctuation)))
    # 20. stopword count
    stop_words = list(set(stopwords.words('english')))
    features_df['20a'] = x_train_df['Input.threadtitle'].apply(lambda x: len([s for s in x.split() if x.lower in stop_words]))
    features_df['20b'] = x_train_df['Posts'].apply(lambda x: len([s for s in x.split() if x.lower in stop_words]))
    # 21. text polarity
    features_df['21a'] = x_train_df['Input.threadtitle'].apply(lambda x: TextBlob(x).sentiment.polarity)
    features_df['21b'] = x_train_df['Posts'].apply(lambda x: TextBlob(x).sentiment.polarity)
    # 22. text subjectivity
    features_df['22a'] = x_train_df['Input.threadtitle'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    features_df['22b'] = x_train_df['Posts'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    # 23. noun count
    features_df['23a'] = x_train_df['Input.threadtitle'].apply(lambda x: count_partofspeech_tag(text=x, tag='noun'))
    features_df['23b'] = x_train_df['Posts'].apply(lambda x: count_partofspeech_tag(text=x, tag='noun'))
    # 24. verb count
    features_df['24a'] = x_train_df['Input.threadtitle'].apply(lambda x: count_partofspeech_tag(text=x, tag='verb'))
    features_df['24b'] = x_train_df['Posts'].apply(lambda x: count_partofspeech_tag(text=x, tag='verb'))
    # 25. adjective count
    features_df['25a'] = x_train_df['Input.threadtitle'].apply(lambda x: count_partofspeech_tag(text=x, tag='adj'))
    features_df['25b'] = x_train_df['Posts'].apply(lambda x: count_partofspeech_tag(text=x, tag='adj'))
    # 26. adverb count
    features_df['26a'] = x_train_df['Input.threadtitle'].apply(lambda x: count_partofspeech_tag(text=x, tag='adv'))
    features_df['26b'] = x_train_df['Posts'].apply(lambda x: count_partofspeech_tag(text=x, tag='adv'))
    # 27. pronoun count
    features_df['27a'] = x_train_df['Input.threadtitle'].apply(lambda x: count_partofspeech_tag(text=x, tag='pronoun'))
    features_df['27b'] = x_train_df['Posts'].apply(lambda x: count_partofspeech_tag(text=x, tag='pronoun'))
    
    # == Post Emission Features == TODO but not sure
    # 28. (pi, hi) = count of occurrences of question words or question marks in pi if the state is hi; 0 otherwise.
    # 29. (pi, hi) = count of occurrences of thank words (thank you or thanks) in pi if the state is hi; 0 otherwise.
    # 30. (pi, hi) = count of occurrences of greeting words (e.g. hi, hello, good morning, welcome etc ) in pi if the state is hi; 0 otherwise.
    # 31. (pi, hi) = count of occurrences of assessment related words (e.g. grade, exam, assignment, quiz, reading, project etc.) in pi if the state is hi; 0 otherwise.
    # 32. (p, ihi) = count of occurrences of request, submit or suggest in pi if the state is hi; 0 otherwise.
    # 33. (pi, hi) = log(course duration/t(pi)) if the state is hi; 0 otherwise. Here t(pi) is the difference between the posting time of pi and the closest course landmark (assignment or project deadline or exam).
    # 34. (pi, pi−1, hi) = difference between posting times of pi and pi−1 normalized by course duration if the state is hi; 0 otherwise.
    
    # == Transition Features == TODO but not sure
    # 35. (hi−1, hi) = 1 if previous state is hi−1 and current state is hi; 0 otherwise.
    # 36. (hi−1, hi, pi, pi−1) = cosine similarity between pi−1 and pi if previous state is hi−1 and current state is hi; 0 otherwise.
    # 37. (hi−1, hi, pi, pi−1) = length of pi if previous state is hi−1, pi−1 has non-zero question words and current state is hi; 0 otherwise.
    # 38. (hn, r) = 1 if last post’s state is hn and intervention decision is r; 0 otherwise.
    # 39. (hn, r, pn) = 1 if last post’s state is hn, pn has non-zero question words and intervention decision is r; 0 otherwise.
    # 40. (hn, r, pn) = log(course duration/t(pn)) if last post’s state is hn and intervention decision is r; 0 otherwise. Here t(pn) is the difference between the posting time of pn and the closest course landmark (assignment or project deadline or exam).

    result = sp.sparse.csr_matrix(features_df.values)
    # result = preprocess(data=result)
    return result


def count_partofspeech_tag(text, tag):
    pos_dict = {'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
                'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                'adj':  ['JJ', 'JJR', 'JJS'],
                'adv': ['RB', 'RBR', 'RBS', 'WRB'],
                'pronoun': ['PRP', 'PRP$', 'WP', 'WP$']}
    count = 0
    for x in TextBlob(text).tags:
        if list(x)[1] in pos_dict[tag]:
            count += 1
    return count


def preprocess(data):
    scaler = MaxAbsScaler()
    # scaler = StandardScaler(with_mean=False)
    return scaler.fit_transform(data)


def main():
    """ load training, validation, and test data """
    dataset_path_1 = r'datasets/Task1-Marking_Task//'
    dataset_path_2 = r'datasets/Task2-Categorisation_Task_low_lvl//'
    dataset_path_3 = r'datasets/Task2-Categorisation_Task_top_lvl//'
    dataset_file = r'_organalysis-003.lecture.5.csv'

    train = pd.read_csv(dataset_path_1 + dataset_file)
    features_df = engineer_features(x_train=train)


# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
